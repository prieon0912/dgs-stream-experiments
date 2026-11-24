"""Stop this Runpod Pod after saved results; never terminate it or print credentials."""
import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--record", default="/workspace/stop_control.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]{8,40}", args.pod_id):
        raise SystemExit("Invalid Pod ID")
    if os.environ.get("RUNPOD_POD_ID") != args.pod_id:
        raise SystemExit("Refusing to control a Pod other than the current Pod")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not configured")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    url = "https://api.runpod.io/v2/pods/" + args.pod_id
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as response:
        pod = json.load(response)
    if pod.get("id") != args.pod_id or "stop" not in pod.get("actions", []):
        raise SystemExit("The current Pod does not expose a stop action")
    record = {"pod_id": args.pod_id, "checked_at": datetime.now(timezone.utc).isoformat(),
              "read_access_verified": True, "stop_action_available": True,
              "write_access_tested": False, "stop_requested": False}
    target = Path(args.record)
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.check_only:
        print("Current-Pod control check passed; no state was changed.", flush=True)
        return
    record.update(stop_requested=True, requested_at=datetime.now(timezone.utc).isoformat())
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.sync()
    print("Saved results; requesting STOP of the current Pod (persistent volume retained).", flush=True)
    request = urllib.request.Request(url + "/action", data=b'{"action":"stop"}',
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        record["stop_response_status"] = response.status
    record["write_access_tested"] = True
    target.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.sync()


if __name__ == "__main__":
    main()
