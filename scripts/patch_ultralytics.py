
from __future__ import annotations

import importlib
from pathlib import Path
import shutil

IMPORTS = (
    "from src.detection.shufflenetv2 import ShuffleV2Stage\n"
    "from src.detection.shuffle_attention import ShuffleAttention\n"
)

CUSTOM_BRANCHES = (
    "        elif m is ShuffleV2Stage:\n"
    "            # Preserve the paper's exact stage width; do not apply\n"
    "            # Ultralytics width scaling to 116/232/464.\n"
    "            c1, c2 = ch[f], args[0]\n"
    "            args = [c1, c2, *args[1:]]\n"
    "        elif m is ShuffleAttention:\n"
    "            # SA preserves the incoming channel count. YAML supplies G.\n"
    "            c1 = ch[f]\n"
    "            c2 = c1\n"
    "            args = [c1, c2, *args]\n"
)

def patch() -> None:
    tasks = importlib.import_module("ultralytics.nn.tasks")
    tasks_path = Path(tasks.__file__)
    text = tasks_path.read_text(encoding="utf-8")

    # Make custom classes available to parse_model's globals()[m].
    if "from src.detection.shufflenetv2 import ShuffleV2Stage" not in text:
        marker = "from ultralytics.utils import ("
        if marker not in text:
            raise RuntimeError(
                "Could not find 'from ultralytics.utils import (' in tasks.py."
            )
        text = text.replace(marker, IMPORTS + "\n" + marker, 1)

    # Do NOT add these modules to base_modules. That would width-scale 116
    # into 120. Use explicit parser branches instead.
    if "elif m is ShuffleV2Stage:" not in text:
        marker = "        elif m is AIFI:"
        if marker not in text:
            raise RuntimeError(
                "Could not find the AIFI parser branch in tasks.py."
            )
        text = text.replace(marker, CUSTOM_BRANCHES + marker, 1)

    backup = tasks_path.with_suffix(tasks_path.suffix + ".milestone1.bak")
    if not backup.exists():
        shutil.copy2(tasks_path, backup)
        print(f"Backup created: {backup}")

    tasks_path.write_text(text, encoding="utf-8")

    updated = tasks_path.read_text(encoding="utf-8")
    required = [
        "from src.detection.shufflenetv2 import ShuffleV2Stage",
        "from src.detection.shuffle_attention import ShuffleAttention",
        "elif m is ShuffleV2Stage:",
        "elif m is ShuffleAttention:",
    ]
    missing = [x for x in required if x not in updated]
    if missing:
        raise RuntimeError(f"Patch verification failed. Missing: {missing}")

    print(f"Patched: {tasks_path}")
    print("Registered custom parser handling:")
    print("  - ShuffleV2Stage")
    print("  - ShuffleAttention")
    print("Custom modules were intentionally NOT added to base_modules.")

if __name__ == "__main__":
    patch()
