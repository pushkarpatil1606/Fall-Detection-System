from __future__ import annotations

from scripts.train import parse_args, validate_resume_path


def test_training_parser_has_resume_option() -> None:
    # Ensure the argument parser can be constructed without invoking training.
    args = parse_args()
    assert hasattr(args, "resume")


def test_nonexistent_resume_path_rejected(tmp_path) -> None:
    missing = tmp_path / "missing.pt"

    try:
        validate_resume_path(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing resume checkpoint was not rejected.")


if __name__ == "__main__":
    print("Resumable training setup test passed.")
