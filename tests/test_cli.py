import subprocess
import sys


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "soundgraft.cli", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--clip" in result.stdout
    assert "--from-clip" in result.stdout
    assert "--it-is-what-it-is" in result.stdout


def test_missing_required_args():
    result = subprocess.run(
        [sys.executable, "-m", "soundgraft.cli"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
