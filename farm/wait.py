#!/usr/bin/env python3
"""Block until no pending/running farm task matches a name pattern.

    farm/wait.py 'bh_*' [--timeout 3600] [--poll 20]

PATTERN is an fnmatch glob on the task name (as given to submit.py). On exit
prints how many matching tasks are done/failed and the log tail of failures.
Exit code: 0 all done, 1 some failed, 2 timeout.
"""
import argparse
import fnmatch
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QROOT = os.environ.get("FARM_QUEUE", os.path.join(HERE, "queue"))


def _names(d):
    out = []
    for e in os.scandir(d):
        if e.is_dir():
            out.extend(_names(e.path))
        elif e.name.endswith(".json") and not e.name.startswith("."):
            out.append(e.name[:-5])
    return out


def _match(stem, pat):
    # file stems are p<prio>_<name>
    name = stem.split("_", 1)[1] if stem[:1] == "p" and "_" in stem else stem
    return fnmatch.fnmatch(name, pat)


def scan(pat):
    res = {}
    for k in ("pending", "running", "done", "failed"):
        res[k] = [n for n in _names(os.path.join(QROOT, k)) if _match(n, pat)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--poll", type=float, default=20)
    ap.add_argument("--tail", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        s = scan(a.pattern)
        if not s["pending"] and not s["running"]:
            break
        if time.time() - t0 > a.timeout:
            print(f"TIMEOUT pending={len(s['pending'])} running={len(s['running'])} "
                  f"done={len(s['done'])} failed={len(s['failed'])}")
            raise SystemExit(2)
        time.sleep(a.poll)
    print(f"done={len(s['done'])} failed={len(s['failed'])} ({time.time() - t0:.0f}s)")
    for n in sorted(s["failed"]):
        print(f"--- FAILED {n}")
        log = os.path.join(QROOT, "logs", n + ".log")
        if os.path.exists(log):
            with open(log, errors="replace") as fh:
                print("".join(fh.readlines()[-a.tail:]))
    raise SystemExit(1 if s["failed"] else 0)


if __name__ == "__main__":
    main()
