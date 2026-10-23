#!/usr/bin/env python3
"""Connect (or disconnect) Tello drones in parallel and verify each one.

  python connect.py 5 8              # connect drones 5 and 8
  python connect.py 1 2 3 4 5 8 9    # any list
  python connect.py --all            # every drone with an ssid in dongles.json
  python connect.py 5 8 --down       # disconnect them

Uses the NetworkManager profiles named TELLO-<n> (created here if missing, always pinned
to that drone's USB dongle from dongles.json, never the laptop WiFi), retries
each drone up to 3 times (rescanning its adapter in between), then prints a table:
adapter, SSID, IP. Exit code 0 only if every requested drone is connected.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, timeout=40):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "timeout"


def iface_ip(iface):
    _, out = run(["ip", "-j", "-4", "addr", "show", iface], 5)
    try:
        for dev in json.loads(out or "[]"):
            for a in dev.get("addr_info", []):
                if a.get("local", "").startswith("192.168.10."):
                    return a["local"]
    except json.JSONDecodeError:
        pass
    return None


def active_ssid(iface):
    _, out = run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list", "ifname", iface,
                  "--rescan", "no"], 10)
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


def connect(n, entry, result, tries=3):
    iface, ssid, prof = entry["iface"], entry["ssid"], f"TELLO-{n}"
    msg = ""
    if not iface.startswith("wlx"):
        result[n] = (False, f"refusing: {iface!r} is not a USB dongle (laptop WiFi is never used)")
        return
    code, _ = run(["nmcli", "-g", "connection.id", "con", "show", prof], 10)
    if code != 0:
        # Open network, pinned to this drone's dongle, never auto-joined.
        code, msg = run(["nmcli", "con", "add", "type", "wifi", "ifname", iface, "con-name", prof,
                         "ssid", ssid, "connection.autoconnect", "no"], 10)
        if code != 0:
            result[n] = (False, f"could not create profile '{prof}': {msg.splitlines()[-1] if msg else 'failed'}")
            return
    else:
        run(["nmcli", "con", "modify", prof, "connection.interface-name", iface,
             "connection.autoconnect", "no"], 10)
    for t in range(1, tries + 1):
        code, msg = run(["nmcli", "con", "up", prof, "ifname", iface])
        if code == 0:
            break
        run(["nmcli", "dev", "wifi", "rescan", "ifname", iface], 10)
        time.sleep(2)
    ip, cur = iface_ip(iface), active_ssid(iface)
    if ip and cur == ssid:
        result[n] = (True, f"{iface}  {cur}  {ip}")
    elif ip:
        result[n] = (False, f"{iface} connected to {cur!r}, expected {ssid!r}")
    else:
        short = msg.splitlines()[-1] if msg else "failed"
        if "could not be found" in msg:
            short = "drone not broadcasting (off / asleep / flat battery)"
        result[n] = (False, short)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drones", nargs="*", type=int, help="drone numbers, e.g. 5 8")
    ap.add_argument("--all", action="store_true", help="every drone with an ssid in dongles.json")
    ap.add_argument("--down", action="store_true", help="disconnect instead")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "dongles.json") if os.path.exists(os.path.join(HERE, "..", "config", "dongles.json")) else os.path.join(HERE, "dongles.json"))
    args = ap.parse_args()

    with open(args.config) as f:
        data = json.load(f)
    data = data.get("dongles", []) if isinstance(data, dict) else data
    table = {int(e["number"]): e for e in data if e.get("ssid")}
    nums = sorted(table) if args.all else sorted(set(args.drones))
    if not nums:
        sys.exit("give drone numbers (e.g. python connect.py 5 8) or --all")
    missing = [n for n in nums if n not in table]
    if missing:
        sys.exit(f"no ssid for drone(s) {missing} in {args.config}")

    if args.down:
        ths = [threading.Thread(target=run, args=(["nmcli", "con", "down", f"TELLO-{n}"],))
               for n in nums]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        print(f"disconnected: {nums}")
        return 0

    print(f"connecting {nums} in parallel ...")
    result = {}
    ths = [threading.Thread(target=connect, args=(n, table[n], result)) for n in nums]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    ok = [n for n in nums if result.get(n, (False,))[0]]
    for n in nums:
        good, info = result.get(n, (False, "no result"))
        print(f"  {'OK  ' if good else 'FAIL'}  drone {n}: {info}")
    print(f"{len(ok)}/{len(nums)} connected" + ("" if len(ok) == len(nums) else
          f"  -> retry the failed ones:  python connect.py {' '.join(str(n) for n in nums if n not in ok)}"))
    return 0 if len(ok) == len(nums) else 1


if __name__ == "__main__":
    sys.exit(main())
