import hashlib
import subprocess
from pathlib import Path


def repo_root() -> Path:
    cmd = ["git", "rev-parse", "--show-toplevel"]
    root = subprocess.check_output(cmd, text=True).strip()
    return Path(root)


def get_conf_hash(conf_stem: str) -> str:
    conf_file = repo_root() / "configs" / f"{conf_stem}.yaml"
    conf_text = conf_file.read_text()
    return hashlib.sha256(conf_text.encode()).hexdigest()[:6]
