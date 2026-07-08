#!/usr/bin/env python
"""Run confidence-bin and edit-transition analysis for CoDraft-Diff.

Inputs:
  - configs/codraft_diff_wikitext2_gpu1.yaml
  - results/codraft_diff_gpu1/draft_refine_adapter/checkpoint.pt
Output:
  - results/codraft_diff_gpu1_extended/analysis/
GPU:
  - Use CUDA_VISIBLE_DEVICES externally, e.g. CUDA_VISIBLE_DEVICES=1.
Resume:
  - Existing files are overwritten only when this script is rerun explicitly.
"""

from codraft_extended_eval import main


if __name__ == "__main__":
    import sys

    sys.argv = [
        sys.argv[0],
        "analysis",
        *sys.argv[1:],
    ]
    main()
