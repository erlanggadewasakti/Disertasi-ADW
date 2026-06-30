"""
run_seed_experiments.py
-----------------------
Execute model training across multiple seeds and aggregate results.

Usage:
  python run_seed_experiments.py                  # Run ALL models x ALL seeds (hours!)
  python run_seed_experiments.py --smoke          # 1 model x 1 seed  (quick test)
  python run_seed_experiments.py --model lstm     # Single model x all seeds
  python run_seed_experiments.py --seeds 42,123   # Custom seeds
  python run_seed_experiments.py --summary-only   # Just aggregate existing results
"""

import argparse, csv, os, sys, datetime, time, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = "D:/MVSA_SINGLE/results"
RES_CSV  = os.path.join(RES_DIR, "seed_runs.csv")
SUMMARY_CSV = os.path.join(RES_DIR, "seed_summary.csv")

SEEDS = [42, 123, 2024, 7, 100]
MODELS = {
    "lstm":        "mvsa-lstm.ipynb",
    "bilstm":      "mvsa-bilstm.ipynb",
    "bert_lstm":   "mvsa-bert-lstm.ipynb",
    "bert_bilstm": "mvsa-bert-bilstm.ipynb",
    "vit_lstm":    "mvsa-vit-lstm.ipynb",
    "vit_bilstm":  "mvsa-vit-bilstm.ipynb",
    "vilbert":     "mvsa-vilbert.ipynb",
}


def run_one(model_key, nb_path, seed):
    """Execute notebook with one seed via papermill."""
    out_path = os.path.join(RES_DIR, f"_exec_{model_key}_s{seed}.ipynb")
    os.makedirs(RES_DIR, exist_ok=True)

    import papermill as pm
    pm.execute_notebook(
        input_path=nb_path,
        output_path=out_path,
        parameters={"seed": seed},
        progress_bar=False,
        log_output=True,
        stdout_file=sys.stdout,
        stderr_file=sys.stderr,
    )
    return out_path


def run_manual_mode(args):
    """Print instructions for running notebooks manually in Jupyter/Colab,
    then aggregate results when ready."""
    print("MANUAL MODE")
    print("=" * 70)
    print("For each model notebook, edit the first cell to change the seed,")
    print("then Run All.  Repeat for each seed in", SEEDS)
    print()
    print("Cell to edit (top of notebook, tagged 'parameters'):")
    print("  seed = 42   <-- change this number")
    print()
    for m, path in MODELS.items():
        print(f"  {m:<12}  {os.path.join(BASE_DIR, path)}")
    print()
    print(f"Results auto-save to:  {RES_CSV}")
    print(f"Then run:  python {__file__} --summary-only")
    print()


def aggregate_results():
    """Read seed_runs.csv, compute mean±std per model, print + save summary."""
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 65)
    print("  AGGREGATED RESULTS  (mean ± sample std, ddof=1)")
    print("=" * 65)

    if not os.path.isfile(RES_CSV):
        print(f"\n  No results file found at {RES_CSV}")
        print("  Run notebooks first or execute seed experiments.")
        return

    df = pd.read_csv(RES_CSV)
    print(f"\n  Loaded {len(df)} rows from {RES_CSV}")
    print(df.to_string(index=False))

    summary = df.groupby("model").agg(
        n=("test_acc", "count"),
        acc_mean=("test_acc", "mean"),
        acc_std=("test_acc", lambda x: x.std(ddof=1)),
        f1_mean=("test_f1", "mean"),
        f1_std=("test_f1", lambda x: x.std(ddof=1)),
    ).reset_index()

    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    summary["f1_std"]  = summary["f1_std"].fillna(0.0)

    print("\n  Summary Table")
    print("  " + "-" * 55)
    header = f"  {'Model':<14} {'N':>3}  {'Acc ± Std':>20}  {'F1 ± Std':>20}"
    print(header)
    print("  " + "-" * 55)

    for _, r in summary.iterrows():
        print(f"  {r['model']:<14} {int(r['n']):>3}  "
              f"{r['acc_mean']:.4f}  ±  {r['acc_std']:.4f}   "
              f"{r['f1_mean']:.4f}  ±  {r['f1_std']:.4f}")

    print("  " + "-" * 55)

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Summary saved to: {SUMMARY_CSV}")

    # LaTeX table
    print("\n  LaTeX Table:")
    print()
    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Model & Accuracy & Macro F1 \\\\\n"
        "\\midrule\n"
    )
    for _, r in summary.iterrows():
        latex += (
            f"  {r['model'].replace('_', ' ').title()} & "
            f"${r['acc_mean']:.4f} \\pm {r['acc_std']:.4f}$ & "
            f"${r['f1_mean']:.4f} \\pm {r['f1_std']:.4f}$ \\\\\n"
        )
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Model performance (mean $\\pm$ std over 5 seeds)}\n"
        "\\label{tab:results}\n"
        "\\end{table}"
    )
    print(latex)

    # MD table
    md_path = os.path.join(RES_DIR, "seed_summary.md")
    md = (
        "# Seed Sweep Results\n\n"
        f"Seeds: {SEEDS}\n\n"
        "| Model | N | Accuracy | Macro F1 |\n"
        "|-------|---|----------|----------|\n"
    )
    for _, r in summary.iterrows():
        md += (f"| {r['model']} | {int(r['n'])} | "
               f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f} | "
               f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} |\n")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n  Markdown summary saved to: {md_path}")
    print()


# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed model experiment runner")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 1 model x 1 seed as a quick smoke test")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only a specific model (e.g. lstm)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (e.g. 42,123,2024)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only aggregate existing CSV; do not run training")
    parser.add_argument("--manual", action="store_true",
                        help="Print manual-run instructions instead of executing")
    args = parser.parse_args()

    if args.summary_only:
        aggregate_results()
        sys.exit(0)

    if args.manual:
        run_manual_mode(args)
        sys.exit(0)

    # Determine run set
    seeds_to_run = SEEDS
    if args.seeds:
        seeds_to_run = [int(s.strip()) for s in args.seeds.split(",")]

    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model '{args.model}'. Choose from: {list(MODELS)}")
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}

    if args.smoke:
        first = list(models_to_run.keys())[0]
        models_to_run = {first: models_to_run[first]}
        seeds_to_run = [seeds_to_run[0]]
        print(f"[SMOKE TEST]  {first}  ×  seed={seeds_to_run[0]}")

    total_runs = len(models_to_run) * len(seeds_to_run)
    print(f"\nTotal runs: {len(models_to_run)} models × {len(seeds_to_run)} seeds = {total_runs}")
    print(f"Results CSV: {RES_CSV}\n")

    # Check papermill availability
    try:
        import papermill as pm
    except ImportError:
        print("ERROR: papermill not installed.  Run:  pip install papermill")
        print("\nSwitching to manual mode instructions...")
        run_manual_mode(args)
        sys.exit(1)

    run_num = 0
    failures = []
    t_start = time.time()

    for model_key in models_to_run:
        nb_path = os.path.join(BASE_DIR, models_to_run[model_key])
        if not os.path.isfile(nb_path):
            print(f"  SKIP  {model_key}  (file not found: {nb_path})")
            continue

        for seed in seeds_to_run:
            run_num += 1
            tag = f"[{run_num}/{total_runs}]"
            print(f"\n{'='*60}\n{tag}  {model_key}  seed={seed}\n{'='*60}")
            t0 = time.time()
            try:
                out = run_one(model_key, nb_path, seed)
                elapsed = time.time() - t0
                print(f"{tag}  DONE  {model_key}  s={seed}  "
                      f"({elapsed/60:.1f} min)  -> {out}")
            except Exception as exc:
                elapsed = time.time() - t0
                failures.append((model_key, seed, str(exc)))
                print(f"{tag}  FAILED  {model_key}  s={seed}  "
                      f"({elapsed/60:.1f} min)")
                traceback.print_exc()

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All done.  Total time: {total_elapsed/60:.1f} min  "
          f"({total_elapsed/3600:.1f} hr)")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for m, s, err in failures:
            print(f"  {m}  seed={s}  : {err[:120]}")

    # Auto-aggregate
    aggregate_results()
