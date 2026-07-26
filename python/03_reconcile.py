"""Verify the R and Python shot-quality implementations agree.

Same discipline as the other studies: two independent implementations of the
same definitions, compared numerically, non-zero exit on mismatch so this can
gate a commit. The model is deterministic in both languages (same IRLS, same
stopping rule), so coefficients are compared to 1e-6 — not "directionally".

Run:  python python/03_reconcile.py   (after 02_model.py and R/02_model.R)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# (python file, r file, join keys, numeric cols, tolerance)
TABLES = [
    ("coefficients.csv", "coefficients_r.csv", ["term"], ["coef"], 1e-6),
    ("calibration.csv", "calibration_r.csv", ["decile"],
     ["n", "mean_pred", "actual"], 1e-6),
    ("player_table.csv", "player_table_r.csv", ["PLAYER_ID"],
     ["fga", "pts", "xpts", "pps", "xpps", "making_100", "making_100_se"], 1e-6),
    ("clock_decomposition.csv", "clock_decomposition_r.csv", ["bucket"],
     ["n", "pps", "xpps", "execution_100"], 1e-6),
]


def compare(py_name: str, r_name: str, keys: list[str], cols: list[str],
            tol: float) -> tuple[bool, str]:
    py = pl.read_csv(OUT / py_name, null_values=["NA"])
    r = pl.read_csv(OUT / r_name, null_values=["NA"])
    merged = py.join(r, on=keys, how="full", suffix="_r", coalesce=True)
    if merged.height != py.height or py.height != r.height:
        return False, (f"row counts differ: py={py.height} r={r.height} "
                       f"joined={merged.height}")
    worst = 0.0
    for c in cols:
        a, b = merged[c], merged[f"{c}_r"]
        # Nulls must agree (e.g. SE undefined for 1-FGA players in BOTH
        # languages); then compare where defined.
        if (a.is_null() != b.is_null()).any():
            return False, f"column {c}: null pattern differs"
        d = (a - b).abs().max()
        worst = max(worst, 0.0 if d is None else d)
    return worst <= tol, f"{py.height} rows, max abs diff {worst:.2e}"


def main() -> int:
    print("Reconciling R and Python shot-quality implementations\n")
    all_ok = True
    for py_name, r_name, keys, cols, tol in TABLES:
        ok, msg = compare(py_name, r_name, keys, cols, tol)
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {py_name:28s} {msg}")
    print("\n" + ("ALL CHECKS PASS" if all_ok else "MISMATCHES FOUND"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
