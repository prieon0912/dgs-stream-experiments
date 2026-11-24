"""Stable fingerprints of measured environments, independent of paths and timestamps."""
import hashlib
import json


def environment_fingerprint(data):
    if not data:
        return None
    repositories = {name: dict(commit=repo.get("commit"), changes=repo.get("changes"))
                    for name, repo in data.get("repositories", {}).items()}
    selected = dict(python=data.get("python"), packages=data.get("packages"), direct_urls=data.get("direct_urls"),
                    repositories=repositories, gpu=data.get("gpu"), nvcc=data.get("nvcc"),
                    experiment_code_sha256=data.get("experiment_code_sha256"))
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
