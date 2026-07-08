#!/usr/bin/env python
"""Regenerate CoDraft-Diff plots from saved extended CSV files.

This script is intentionally light: the main plotting is performed during
`codraft_extended_eval.py` runs. Re-run the corresponding subcommand with
FORCE=1 if source CSV values change.
"""

from pathlib import Path


def main() -> None:
    out_dir = Path("results/codraft_diff_gpu1_extended")
    expected = [
        out_dir / "ratio_sweep" / "accuracy_vs_refine_ratio.png",
        out_dir / "ratio_sweep" / "wrong_edit_rate_vs_refine_ratio.png",
        out_dir / "ratio_sweep" / "correction_recall_vs_refine_ratio.png",
        out_dir / "ratio_sweep" / "precision_recall_tradeoff.png",
        out_dir / "analysis" / "confidence_bin_error_rate.png",
        out_dir / "analysis" / "confidence_bin_edit_quality.png",
        out_dir / "analysis" / "edit_transition_stacked_bar.png",
    ]
    for path in expected:
        print(f"{path}: {'exists' if path.exists() else 'missing'}")


if __name__ == "__main__":
    main()
