import shutil
from pathlib import Path

# Absolute path to the training dataset (outside the repo)
train_src = Path(r"D:\ML_Hackathon\Student folder\train")
# Fixtures directory inside the repository
repo_root = Path(__file__).resolve().parents[1]
fixture_dir = repo_root / "tests" / "fixtures"
fixture_dir.mkdir(parents=True, exist_ok=True)

# Choose at least five distinct trips for GroupKFold
trip_ids = ["00000001", "00000002", "00000003", "00000004", "00000005"]
for trip in trip_ids:
    src = train_src / trip
    dst = fixture_dir / trip
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
print(f"Copied {len(trip_ids)} trips to fixture directory: {fixture_dir}")
