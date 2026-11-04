#!/usr/bin/env python3
"""Node-level task-farm manager.

One manager runs per compute node. It claims task files from a shared queue
directory (atomic rename), runs each task pinned to its own physical cores and
records the outcome. Tasks can be added at any time with ``farm/submit.py``.

Queue layout (``$FARM_QUEUE``, default ``farm/queue``)::

    pending/<name>.json          waiting tasks: lowest priority number first; inside a
                                 priority band round-robin over owners (task-name prefix
                                 up to the second '-', e.g. ``dev-race``), FIFO per owner
    running/<host>/<name>.json   claimed by the manager on <host>
    done/<name>.json             finished with exit code 0 (+ run metadata)
    failed/<name>.json           non-zero exit code or timeout
    logs/<name>.log              stdout/stderr of the task
    hosts/<host>.json            heartbeat of each manager
    control/DRAIN                stop claiming, exit when idle
    control/STOP                 requeue running tasks and exit now

Task file keys: ``cmd`` (str for a shell command or list of argv), optional
``cwd``, ``ncores`` (default 1), ``timeout`` (s), ``env`` (dict),
``max_retries`` (default 1).
"""
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QROOT = os.environ.get("FARM_QUEUE", os.path.join(HERE, "queue"))
D = {k: os.path.join(QROOT, k) for k in
     ("pending", "running", "done", "failed", "logs", "hosts", "control")}
HOST = socket.gethostname()
JOBID = os.environ.get("SLURM_JOB_ID", "local")
NCORES = int(os.environ.get("FARM_CORES", "128"))
POLL = float(os.environ.get("FARM_POLL", "3"))
HEARTBEAT_EVERY = 30.0
MY_RUNNING = os.path.join(D["running"], HOST)


def physical_cpus(n):
    """One logical CPU per physical core, restricted to the CPUs we may use."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, primary = set(), []
    for c in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                sib = f.read().strip()
        except OSError:
            sib = str(c)
        if sib not in seen:
            seen.add(sib)
            primary.append(c)
    return primary[:n]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {HOST} {msg}", flush=True)


def atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{HOST}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


class Manager:
    def __init__(self):
        for d in D.values():
            os.makedirs(d, exist_ok=True)
        os.makedirs(MY_RUNNING, exist_ok=True)
        self.cpus = physical_cpus(NCORES)
        self.ncores = len(self.cpus)
        self.free = set(self.cpus)
        self.procs = {}  # name -> dict(proc, cpus, start, task, path, logf)
        self.stop_requested = False
        self.last_hb = 0.0
        self.n_started = 0
        self.n_finished = 0
        self.t0 = time.time()
        self.backoff_until = 0.0
        self._mt = {}
        signal.signal(signal.SIGUSR1, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        self._recover_own_leftovers()

    def _on_signal(self, signum, frame):
        log(f"received signal {signum}; requeueing running tasks")
        self.stop_requested = True

    def _recover_own_leftovers(self):
        for name in os.listdir(MY_RUNNING):
            if name.endswith(".json"):
                try:
                    os.rename(os.path.join(MY_RUNNING, name), os.path.join(D["pending"], name))
                    log(f"recovered leftover task {name}")
                except OSError:
                    pass

    # ------------------------------------------------------------------ claim
    def _mtime(self, name):
        t = self._mt.get(name)
        if t is None:
            try:
                t = os.stat(os.path.join(D["pending"], name)).st_mtime
            except OSError:
                t = 0.0
            self._mt[name] = t
        return t

    def _order(self, names):
        groups = {}
        for n in names:
            prio, _, rest = n.partition("_")
            owner = "-".join(rest.split("-")[:2])
            groups.setdefault((prio, owner), []).append(n)
        keyed = []
        # ties between owners at the same rank are broken randomly each poll, so a node
        # with few free cores does not always serve the alphabetically first owners
        rnd = {key: random.random() for key in groups}
        for (prio, owner), lst in groups.items():
            lst.sort(key=lambda n: (self._mtime(n), n))
            keyed.extend((prio, rank, rnd[(prio, owner)], n) for rank, n in enumerate(lst))
        keyed.sort()
        self._mt = {n: self._mt[n] for n in names if n in self._mt}
        return [k[-1] for k in keyed]

    def _claim(self):
        if not self.free or time.time() < self.backoff_until:
            return
        try:
            names = [e.name for e in os.scandir(D["pending"])
                     if e.name.endswith(".json") and e.is_file()]
        except FileNotFoundError:
            return
        if not names:
            return
        names = self._order(names)
        window = names[: max(4 * len(self.free), 16)]
        # light shuffle inside a priority band to reduce claim collisions between nodes
        head = window[: len(self.free) * 2]
        random.shuffle(head)
        window = head + window[len(head):]
        for name in window:
            if not self.free:
                break
            src = os.path.join(D["pending"], name)
            dst = os.path.join(MY_RUNNING, name)
            try:
                os.rename(src, dst)
            except OSError:
                continue  # claimed by someone else
            try:
                task = read_json(dst)
            except Exception as exc:  # malformed task file
                log(f"bad task file {name}: {exc}")
                os.replace(dst, os.path.join(D["failed"], name))
                continue
            need = max(1, int(task.get("ncores", 1)))
            if need > len(self.free):
                if need > self.ncores:
                    task["farm_error"] = f"ncores={need} exceeds node size {self.ncores}"
                    atomic_write_json(os.path.join(D["failed"], name), task)
                    os.remove(dst)
                else:
                    os.rename(dst, src)  # give it back; another node may have room
                continue
            if not self._launch(name, dst, task, need):
                try:
                    os.rename(dst, src)
                except OSError:
                    pass
                self.backoff_until = time.time() + 60
                break

    def _launch(self, name, path, task, need):
        """Start a task; False (nothing changed) if its log cannot be opened."""
        logpath = os.path.join(D["logs"], name[:-5] + ".log")
        try:
            logf = open(logpath, "ab")
            logf.write(f"# host={HOST} job={JOBID} start={time.ctime()}\n".encode())
            logf.flush()
        except OSError as exc:
            log(f"cannot open log for {name}: {exc}; backing off")
            return False
        cpus = sorted(self.free)[:need]
        for c in cpus:
            self.free.discard(c)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in task.get("env", {}).items()})
        env["OMP_NUM_THREADS"] = str(task.get("env", {}).get("OMP_NUM_THREADS", need))
        env["FARM_TASK"] = name
        env["FARM_CPUS"] = ",".join(map(str, cpus))
        cmd = task["cmd"]

        def pin():
            try:
                os.sched_setaffinity(0, set(cpus))
            except OSError:
                pass

        try:
            proc = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=task.get("cwd") or None,
                                    stdout=logf, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True, preexec_fn=pin)
        except Exception as exc:
            logf.close()
            for c in cpus:
                self.free.add(c)
            task["farm_error"] = f"launch failed: {exc}"
            try:
                atomic_write_json(os.path.join(D["failed"], name), task)
                os.remove(path)
            except OSError:
                return False
            return True
        self.procs[name] = dict(proc=proc, cpus=cpus, start=time.time(), task=task,
                                path=path, logf=logf, reason=None)
        self.n_started += 1
        return True

    # ------------------------------------------------------------------ reap
    def _kill(self, rec):
        try:
            os.killpg(rec["proc"].pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(20):
            if rec["proc"].poll() is not None:
                return
            time.sleep(0.1)
        try:
            os.killpg(rec["proc"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _finish(self, name, rec, rc, reason=None):
        """Record the outcome; raises OSError (state unchanged) if it cannot be written."""
        task = rec["task"]
        end = time.time()
        hist = list(task.get("farm_history", [])) + [
            dict(host=HOST, job=JOBID, start=rec["start"], end=end,
                 elapsed=round(end - rec["start"], 3), rc=rc, reason=reason)]
        if rc == 0:
            dest = D["done"]
        elif reason != "timeout" and len(hist) <= int(task.get("max_retries", 1)):
            dest = D["pending"]
        else:
            dest = D["failed"]
        atomic_write_json(os.path.join(dest, name), dict(task, farm_history=hist))
        try:
            os.remove(rec["path"])
        except FileNotFoundError:
            pass
        rec["logf"].close()
        for c in rec["cpus"]:
            self.free.add(c)
        self.n_finished += 1

    def _reap(self):
        now = time.time()
        for name in list(self.procs):
            rec = self.procs[name]
            rc = rec["proc"].poll()
            if rc is None:
                tmo = rec["task"].get("timeout")
                if not (tmo and now - rec["start"] > float(tmo)):
                    continue
                self._kill(rec)
                rec["reason"], rec["rc"] = "timeout", -9
            try:
                self._finish(name, rec, rec.get("rc", rc), reason=rec["reason"])
            except OSError as exc:
                log(f"cannot record outcome of {name}: {exc}; will retry")
                continue
            del self.procs[name]

    def _requeue_all(self):
        for name, rec in list(self.procs.items()):
            self._kill(rec)
            task = rec["task"]
            task["farm_requeued"] = int(task.get("farm_requeued", 0)) + 1
            try:
                atomic_write_json(os.path.join(D["pending"], name), task)
                os.remove(rec["path"])
            except OSError as exc:
                log(f"cannot requeue {name}: {exc} (requeue_stale.py will recover it)")
            rec["logf"].close()
            for c in rec["cpus"]:
                self.free.add(c)
            del self.procs[name]
            log(f"requeued {name}")

    # ------------------------------------------------------------------ main
    def _heartbeat(self, force=False):
        now = time.time()
        if not force and now - self.last_hb < HEARTBEAT_EVERY:
            return
        self.last_hb = now
        try:
            atomic_write_json(os.path.join(D["hosts"], HOST + ".json"), dict(
                host=HOST, job=JOBID, pid=os.getpid(), time=now, ctime=time.ctime(now),
                uptime=round(now - self.t0), ncores=self.ncores, busy=self.ncores - len(self.free),
                running=sorted(self.procs), started=self.n_started, finished=self.n_finished,
                stopping=self.stop_requested))
        except OSError as exc:
            log(f"heartbeat write failed: {exc}")

    def _step(self):
        """One poll cycle; returns False when the manager should exit."""
        self._reap()
        ctrl = os.listdir(D["control"])
        if "STOP" in ctrl:
            self.stop_requested = True
        if self.stop_requested:
            self._requeue_all()
            self._heartbeat(force=True)
            log("stopped")
            return False
        if "DRAIN" not in ctrl:
            self._claim()
        elif not self.procs:
            self._heartbeat(force=True)
            log("drained; exiting")
            return False
        self._heartbeat()
        return True

    def run(self):
        log(f"manager up: job={JOBID} cores={self.ncores} cpus={self.cpus[0]}..{self.cpus[-1]} queue={QROOT}")
        while True:
            try:
                if not self._step():
                    return
            except Exception as exc:  # never let a filesystem hiccup kill the node
                log(f"poll cycle error: {exc!r}; retrying in 30 s")
                if self.stop_requested:
                    self._requeue_all()
                    return
                time.sleep(30)
            time.sleep(POLL)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    Manager().run()
