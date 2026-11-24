"""[any env, stdlib only] Bookkeeping when EDGS gives fewer visible points than the target N.

  python alt_init/edgs_short.py exclude <condition_dir>   # mark this N as excluded (writes excluded.json)
  python alt_init/edgs_short.py best <init_root> <tag>    # print the excluded run with the most visible points
                                                          # (its init_ALL.ply becomes condition edgs_all)
"""
import json
import sys
from pathlib import Path


def exclude(d):
    d = Path(d)
    r = json.load(open(d / "init_ALL_edgs.json"))
    json.dump(dict(reason="fewer visible points than N", n_target=r["n_target"], n_visible=r["n_visible"],
                   all_points_ply=str(d / "init_ALL.ply")), open(d / "excluded.json", "w"), indent=1)
    print(f"excluded N={r['n_target']} (visible {r['n_visible']})")


def best(root, tag):
    found = None
    for d in sorted(Path(root).glob(f"{tag}_edgs_*")):
        j = d / "init_ALL_edgs.json"
        if (d / "excluded.json").is_file() and j.is_file() and (d / "init_ALL.ply").is_file():
            n = json.load(open(j))["n_visible"]
            if found is None or n > found[0]:
                found = (n, d)
    return str(found[1]) if found else ""


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "exclude":
        exclude(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "best":
        print(best(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
