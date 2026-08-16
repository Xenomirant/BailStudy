"""Rate-limited git sync of rb_estimator/results to the rb-estimator branch."""
import subprocess
import time

_last_push = [0.0]
MIN_INTERVAL_S = 600
REPO_ROOT = "/workspace/BailStudy"


def sync(msg: str, force: bool = False):
    now = time.time()
    if not force and now - _last_push[0] < MIN_INTERVAL_S:
        return False
    try:
        subprocess.run(["git", "add", "rb_estimator"], cwd=REPO_ROOT, check=True,
                       capture_output=True)
        r = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO_ROOT,
                           capture_output=True)
        if r.returncode == 0:  # something to commit
            subprocess.run(["git", "push", "-q"], cwd=REPO_ROOT, check=True,
                           capture_output=True, timeout=120)
        _last_push[0] = now
        return True
    except Exception as e:  # never let sync kill the run
        print(f"[sync] WARNING: {e}", flush=True)
        return False
