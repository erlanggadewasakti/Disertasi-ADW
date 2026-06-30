"""
Patch all MVSA training notebooks to:
1. Add a papermill-style 'parameters' cell with  seed = 42
2. Replace all  CONFIG["seed"] / CONFIG['seed']  ->  seed
3. Append a cell that writes per-seed results to shared CSV
"""

import json, os, nbformat as nbf

NOTEBOOKS = {
    "mvsa-lstm.ipynb"      : "lstm",
    "mvsa-bilstm.ipynb"    : "bilstm",
    "mvsa-bert-lstm.ipynb" : "bert_lstm",
    "mvsa-bert-bilstm.ipynb": "bert_bilstm",
    "mvsa-vit-lstm.ipynb"  : "vit_lstm",
    "mvsa-vit-bilstm.ipynb": "vit_bilstm",
    "mvsa-vilbert.ipynb"   : "vilbert",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_CSV = r"D:/MVSA_SINGLE/results/seed_runs.csv"


def patch_notebook(fname: str, model_name: str) -> None:
    path = os.path.join(BASE_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # 1. Insert parameters cell at position 0
    param_cell = nbf.v4.new_code_cell("seed = 42")
    param_cell.metadata.tags = ["parameters"]
    nb.cells.insert(0, param_cell)

    # 2. Replace CONFIG["seed"] / CONFIG['seed']  ->  seed  in every code cell
    #    Also change  "seed": 42  ->  "seed": seed  in CONFIG dict definitions
    import re
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = "".join(cell.source)
        modified = False

        # Replace access patterns: CONFIG["seed"] / CONFIG['seed'] -> seed
        if 'CONFIG["seed"]' in text or "CONFIG['seed']" in text:
            text = text.replace('CONFIG["seed"]', "seed")
            text = text.replace("CONFIG['seed']", "seed")
            modified = True

        # Replace dict definition: "seed": 42 -> "seed": seed
        new_text = re.sub(r'("seed"\s*:\s*)\d+', r'\g<1>seed', text)
        if new_text != text:
            text = new_text
            modified = True

        if not modified:
            continue
        lines = text.split("\n")
        new_source = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                new_source.append(line + "\n")
            elif line:  # non-empty
                new_source.append(line)
            # skip trailing empty line
        cell.source = new_source

    # 3. Append results cell at the end
    results_cell_code = f'''# ---- Seed-sweep results recorder ----
import csv, os, datetime

MODEL_NAME = "{model_name}"
os.makedirs(os.path.dirname(r"{RES_CSV}"), exist_ok=True)

# Handle both variable naming conventions
try:
    _r_test_acc = test_metrics["acc"] if isinstance(test_metrics, dict) else test_acc
    _r_test_f1  = test_metrics["f1"]  if isinstance(test_metrics, dict) else test_f1
except (NameError, KeyError, TypeError):
    _r_test_acc = test_acc
    _r_test_f1  = test_f1

file_exists = os.path.isfile(r"{RES_CSV}")
with open(r"{RES_CSV}", "a", newline="", encoding="utf-8") as fobj:
    writer = csv.writer(fobj)
    if not file_exists:
        writer.writerow(["model", "seed", "test_acc", "test_f1", "best_val_f1", "timestamp"])
    writer.writerow([MODEL_NAME, seed, _r_test_acc, _r_test_f1, best_val_f1,
                     datetime.datetime.now().isoformat()])

print(f"\\n[SEED-SWEEP] Recorded  {{MODEL_NAME}}  seed={{seed}}  "
      f"acc={{_r_test_acc:.4f}}  f1={{_r_test_f1:.4f}}  best_val_f1={{best_val_f1:.4f}}")
'''

    nb.cells.append(nbf.v4.new_code_cell(results_cell_code))

    # 4. Write back
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"  Patched {fname}  (model={model_name})")


# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Patching notebooks for seed sweep ...")
    for nb_file, model_id in NOTEBOOKS.items():
        patch_notebook(nb_file, model_id)
    print("Done. All 7 notebooks patched.")
