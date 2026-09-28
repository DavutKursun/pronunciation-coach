"""Publish the demo to a Hugging Face Space.

Log in once with `hf auth login` (or set HF_TOKEN), then:
    python scripts/deploy_space.py DavutKursun/pronunciation-coach --dry-run
    python scripts/deploy_space.py DavutKursun/pronunciation-coach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ["app.py", "requirements.txt", "packages.txt", "pronunciation/*.py",
            "data/sentences.json", "models/scorer.joblib"]


def files_to_upload() -> list[Path]:
    return sorted({p for pattern in PATTERNS for p in ROOT.glob(pattern) if p.is_file()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="e.g. DavutKursun/pronunciation-coach")
    parser.add_argument("--dry-run", action="store_true", help="only list the files")
    args = parser.parse_args()

    files = files_to_upload()
    print("Files for the Space:")
    for f in files:
        print("  ", f.relative_to(ROOT))
    print("   space/README.md -> README.md")
    if not (ROOT / "models" / "scorer.joblib").exists():
        print("note: models/scorer.joblib not found, the demo will show '% sounds correct' instead of a score")
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(ROOT), repo_id=args.repo_id, repo_type="space",
                      allow_patterns=PATTERNS, commit_message="Deploy Pronunciation Coach")
    api.upload_file(path_or_fileobj=str(ROOT / "space" / "README.md"), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="space")
    print(f"Done: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
