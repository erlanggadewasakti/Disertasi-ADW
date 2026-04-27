"""
Script to standardize CONFIG cells across all MVSA notebooks.
Changes made:
  - mvsa-lstm.ipynb:       epochs 30→20, batch_size 32→16
  - mvsa-bilstm.ipynb:     epochs 30→20, batch_size 32→16
  - mvsa-bert-lstm.ipynb:  epochs 30→20
  - mvsa-bert-bilstm.ipynb: epochs 15→20, early_stop_patience 4→5
  - mvsa-vit-lstm.ipynb:   NO CHANGES (already correct)
  - mvsa-vit-bilstm.ipynb: NO CHANGES (already correct)
  - mvsa-vilbert.ipynb:    NO CHANGES (keep label_smoothing, grad_accum, AMP)
"""

import json
import re
import os
import shutil
from datetime import datetime

NOTEBOOK_DIR = r"e:\Coding\Project\Disertasi"

# Define replacements per notebook
# Each entry: (filename, list of (old_string, new_string) pairs to replace in source lines)
CHANGES = {
    "mvsa-lstm.ipynb": [
        # epochs 30 → 20
        ('"epochs"             : 30,', '"epochs"             : 20,'),
        # batch_size 32 → 16
        ('"batch_size"         : 32,', '"batch_size"         : 16,'),
    ],
    "mvsa-bilstm.ipynb": [
        # epochs 30 → 20
        ('"epochs"             : 30,', '"epochs"             : 20,'),
        # batch_size 32 → 16
        ('"batch_size"         : 32,', '"batch_size"         : 16,'),
    ],
    "mvsa-bert-lstm.ipynb": [
        # epochs 30 → 20
        ('"epochs"             : 30,', '"epochs"             : 20,'),
        # Also handle the variant key name
        ('"epochs": 30,', '"epochs": 20,'),
    ],
    "mvsa-bert-bilstm.ipynb": [
        # epochs 15 → 20
        ('"epochs"             : 15,', '"epochs"             : 20,'),
        # early_stop_patience 4 → 5
        ('"early_stop_patience": 4,', '"early_stop_patience": 5,'),
    ],
}


def find_config_cell(notebook_data):
    """Find the cell index that contains the CONFIG dictionary."""
    for i, cell in enumerate(notebook_data["cells"]):
        if cell["cell_type"] != "code":
            continue
        source_text = "".join(cell["source"])
        if "CONFIG" in source_text and '"epochs"' in source_text:
            return i
    return None


def apply_replacements(cell, replacements):
    """Apply string replacements to a cell's source lines."""
    changes_made = []
    new_source = []
    for line in cell["source"]:
        original_line = line
        for old_str, new_str in replacements:
            if old_str in line:
                line = line.replace(old_str, new_str)
                changes_made.append(f"  '{old_str.strip()}' -> '{new_str.strip()}'")
        new_source.append(line)
    cell["source"] = new_source
    return changes_made


def process_notebook(filename, replacements):
    """Process a single notebook file."""
    filepath = os.path.join(NOTEBOOK_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False

    # Create backup
    backup_path = filepath + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"  -> Backup created: {filename}.bak")

    # Read notebook
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find CONFIG cell
    config_idx = find_config_cell(nb)
    if config_idx is None:
        print(f"  FAIL CONFIG cell not found in {filename}")
        return False

    print(f"  -> Found CONFIG in cell #{config_idx}")

    # Apply replacements
    changes = apply_replacements(nb["cells"][config_idx], replacements)

    if not changes:
        print(f"  ! No matching strings found - check the replacement patterns")
        return False

    for c in changes:
        print(f"  OK {c}")

    # Clear outputs to reduce file size (optional but clean)
    # nb["cells"][config_idx]["outputs"] = []

    # Write back
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True


def verify_notebook(filename, expected_values):
    """Verify config values in a notebook."""
    filepath = os.path.join(NOTEBOOK_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    config_idx = find_config_cell(nb)
    if config_idx is None:
        print(f"  FAIL CONFIG cell not found")
        return False

    source_text = "".join(nb["cells"][config_idx]["source"])
    all_ok = True
    for key, expected in expected_values.items():
        # Search for the key-value pattern
        pattern = rf'"{key}"\s*[:=]\s*{re.escape(str(expected))}'
        if re.search(pattern, source_text):
            print(f"  OK {key} = {expected}")
        else:
            print(f"  FAIL {key} != {expected} (MISMATCH!)")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("  Standardizing MVSA Notebook Configurations")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Apply changes
    for filename, replacements in CHANGES.items():
        print(f"\n{'-' * 50}")
        print(f"Processing: {filename}")
        print(f"{'-' * 50}")
        success = process_notebook(filename, replacements)
        if success:
            print(f"  OK {filename} updated successfully")
        else:
            print(f"  FAIL {filename} update FAILED")

    # Verify ALL notebooks
    print(f"\n{'=' * 60}")
    print("  Verification — Checking standardized values")
    print(f"{'=' * 60}")

    verification = {
        "mvsa-lstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-bilstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-bert-lstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-bert-bilstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-vit-lstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-vit-bilstm.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
        "mvsa-vilbert.ipynb": {"epochs": 20, "batch_size": 16, "early_stop_patience": 5},
    }

    all_pass = True
    for filename, expected in verification.items():
        print(f"\n{filename}:")
        if not verify_notebook(filename, expected):
            all_pass = False

    print(f"\n{'=' * 60}")
    if all_pass:
        print("  OK ALL notebooks verified successfully!")
    else:
        print("  ! Some notebooks have mismatched values - check above")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
