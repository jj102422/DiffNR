#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.report import print_markdown_summary, summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-case CT reconstruction metrics.")
    parser.add_argument("--per_case_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to summarize metrics. Install pandas.") from exc

    args = parse_args()
    rows = pd.read_csv(args.per_case_csv).to_dict("records")
    summary = summarize_metrics(rows, model_order=args.models)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print_markdown_summary(summary)


if __name__ == "__main__":
    main()
