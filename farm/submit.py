#!/usr/bin/env python3
"""Add tasks to the farm queue.

Examples::

    # one shell command
    farm/submit.py --name hello --cmd "echo hi; sleep 5"
    # many tasks, one JSON object per line ({"name": ..., "cmd": ..., ...})
    farm/submit.py --jsonl tasks.jsonl
    # status of the queue and the managers
    farm/submit.py --status
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QROOT = os.environ.get("FARM_QUEUE", os.path.join(HERE, "queue"))
SUBDIRS = ("pending", "running", "done", "failed", "logs", "hosts", "control")
D = {k: os.path.join(QROOT, k) for k in SUBDIRS}


def existing_names():
    names = set()
    for k in ("pending", "done", "failed"):
        names.update(os.listdir(D[k]))
    for h in os.listdir(D["running"]):
        p = os.path.join(D["running"], h)
        if os.path.isdir(p):
            names.update(os.listdir(p))
    return names


def fname(task):
    name = task["name"]
    safe = "".join(c if c.isalnum() or c in "-_.+=" else "_" for c in name)
    return f"p{int(task.get('priority', 50)):02d}_{safe}.json"


def submit(tasks, force=False):
    have = existing_names() if not force else set()
    n = 0
    for t in tasks:
        if "cmd" not in t or "name" not in t:
            raise SystemExit(f"task needs 'name' and 'cmd': {t}")
        f = fname(t)
        if f in have:
            continue
        t.setdefault("submitted", time.time())
        tmp = os.path.join(D["pending"], "." + f + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(t, fh)
        os.replace(tmp, os.path.join(D["pending"], f))
        n += 1
    return n


def status(verbose=False):
    def count(d):
        return sum(1 for e in os.scandir(d) if e.name.endswith(".json"))
    running = 0
    for h in os.listdir(D["running"]):
        p = os.path.join(D["running"], h)
        if os.path.isdir(p):
            running += count(p)
    print(f"pending={count(D['pending'])} running={running} "
          f"done={count(D['done'])} failed={count(D['failed'])}")
    now = time.time()
    busy = tot = 0
    for f in sorted(os.listdir(D["hosts"])):
        if not f.endswith(".json"):
            continue
        try:
            hb = json.load(open(os.path.join(D["hosts"], f)))
        except Exception:
            continue
        age = now - hb["time"]
        alive = age < 120 and not hb.get("stopping")
        if alive:
            busy += hb["busy"]
            tot += hb["ncores"]
        if verbose or alive:
            print(f"  {hb['host']:>10s} job={hb['job']} busy={hb['busy']:3d}/{hb['ncores']} "
                  f"started={hb['started']} finished={hb['finished']} "
                  f"heartbeat_age={age:.0f}s {'ALIVE' if alive else 'dead'}")
    if tot:
        print(f"cores busy: {busy}/{tot}")
    ctrl = os.listdir(D["control"])
    if ctrl:
        print("control flags:", " ".join(sorted(ctrl)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--cmd")
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("--ncores", type=int, default=1)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--priority", type=int, default=50, help="lower runs first (0-99)")
    ap.add_argument("--jsonl", help="file with one task per line")
    ap.add_argument("--force", action="store_true", help="resubmit even if the name exists")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    for d in D.values():
        os.makedirs(d, exist_ok=True)
    if a.status:
        status(a.verbose)
        return
    tasks = []
    if a.jsonl:
        src = sys.stdin if a.jsonl == "-" else open(a.jsonl)
        for line in src:
            line = line.strip()
            if line:
                t = json.loads(line)
                # without a cwd the task would run in the node manager's directory
                t.setdefault("cwd", a.cwd)
                tasks.append(t)
    elif a.cmd and a.name:
        t = dict(name=a.name, cmd=a.cmd, cwd=a.cwd, ncores=a.ncores, priority=a.priority)
        if a.timeout:
            t["timeout"] = a.timeout
        tasks.append(t)
    else:
        ap.error("give --name/--cmd, --jsonl or --status")
    n = submit(tasks, a.force)
    print(f"submitted {n} task(s) ({len(tasks) - n} skipped as duplicates)")


if __name__ == "__main__":
    main()
