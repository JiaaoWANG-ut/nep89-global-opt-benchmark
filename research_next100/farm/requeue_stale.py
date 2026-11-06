#!/usr/bin/env python3
"""Move tasks claimed by dead managers (stale heartbeat) back to pending/."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
QROOT = os.environ.get("FARM_QUEUE", os.path.join(HERE, "queue"))
STALE = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0

now = time.time()
n = 0
for host in os.listdir(os.path.join(QROOT, "running")):
    hdir = os.path.join(QROOT, "running", host)
    hb = os.path.join(QROOT, "hosts", host + ".json")
    try:
        age = now - json.load(open(hb))["time"]
    except Exception:
        age = float("inf")
    if age < STALE:
        continue
    for name in os.listdir(hdir):
        if name.endswith(".json"):
            os.rename(os.path.join(hdir, name), os.path.join(QROOT, "pending", name))
            n += 1
print(f"requeued {n} stale task(s)")
