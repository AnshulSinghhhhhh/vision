import os
import tempfile
import pytest
from unittest.mock import patch
from knobs.run_grid import get_git_commit_hash


def test_get_git_commit_hash_synthetic_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_commit_file = os.path.join(tmp_dir, "GIT_COMMIT")
        with open(fake_commit_file, "w") as f:
            f.write("abcdef1234567890-dirty\n")
            
        # 1. Direct file argument
        val = get_git_commit_hash(git_commit_file=fake_commit_file)
        assert val == "abcdef1234567890-dirty"
        
        # 2. Environment variable fallback when git command fails
        with patch.dict(os.environ, {"KNOBS_GIT_COMMIT_FILE": fake_commit_file}):
            with patch("subprocess.run", side_effect=Exception("Git not found")):
                val_env = get_git_commit_hash()
                assert val_env == "abcdef1234567890-dirty"


def test_get_git_commit_hash_fallback_search():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_commit_file = os.path.join(tmp_dir, "GIT_COMMIT")
        with open(fake_commit_file, "w") as f:
            f.write("testcommit456\n")
            
        import sys
        sys.path.insert(0, tmp_dir)
        try:
            with patch("subprocess.run", side_effect=Exception("Git not found")):
                with patch.dict(os.environ, {}, clear=True):
                    val = get_git_commit_hash()
                    assert val == "testcommit456"
        finally:
            if tmp_dir in sys.path:
                sys.path.remove(tmp_dir)
