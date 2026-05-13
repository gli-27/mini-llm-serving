#!/usr/bin/env python3
"""Generate a markdown metrics report from Locust CSV results.

Usage:
    python scripts/generate_metrics_report.py tests/load/results/

Reads *_stats.csv and *_failures.csv files from the results directory
and outputs a formatted markdown table to stdout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def parse_stats_csv(csv_path: Path) -> dict | None:
    """Parse a Locust *_stats.csv file and return the Aggregated row."""
    if not csv_path.exists():
        return None

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                return row
    return None


def parse_failures_csv(csv_path: Path) -> list[dict]:
    """Parse a Locust *_failures.csv file and return all rows."""
    if not csv_path.exists():
        return []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def format_number(value: str | None) -> str:
    """Format a numeric string for display."""
    if value is None or value == "N/A" or value == "":
        return "—"
    try:
        num = float(value)
        if num == int(num):
            return str(int(num))
        return f"{num:.1f}"
    except ValueError:
        return value


def main() -> None:
    """Generate metrics report from Locust results directory."""
    if len(sys.argv) < 2:
        print("Usage: python generate_metrics_report.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    phases = ["warmup", "ramp", "stress"]

    # Header
    print("# Load Test Results")
    print()
    print("| Phase | Requests | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Errors |")
    print("|-------|----------|-----|----------|----------|----------|--------|")

    for phase in phases:
        stats_path = results_dir / f"{phase}_stats.csv"
        row = parse_stats_csv(stats_path)

        if row is None:
            print(f"| {phase.capitalize()} | — | — | — | — | — | — |")
            continue

        # Locust CSV column names vary by version — try common names
        requests = format_number(row.get("Request Count", row.get("# Requests", "")))
        rps = format_number(row.get("Requests/s", row.get("RPS", "")))
        p50 = format_number(row.get("50%", row.get("Median Response Time", "")))
        p95 = format_number(row.get("95%", ""))
        p99 = format_number(row.get("99%", ""))
        failures = format_number(row.get("Failure Count", row.get("# Failures", "0")))

        print(
            f"| {phase.capitalize()} | {requests} | {rps} | {p50} | {p95} | {p99} | {failures} |"
        )

    # Failures summary
    print()
    print("### Error Summary")
    print()

    has_failures = False
    for phase in phases:
        failures_path = results_dir / f"{phase}_failures.csv"
        failures = parse_failures_csv(failures_path)

        if failures:
            has_failures = True
            print(f"**{phase.capitalize()}:**")
            for f in failures:
                method = f.get("Method", "?")
                name = f.get("Name", "?")
                count = f.get("Occurrences", f.get("# Occurrences", "?"))
                msg = f.get("Error", f.get("Message", "?"))
                print(f"- `{method} {name}`: {count}× — {msg}")
            print()

    if not has_failures:
        print("No errors recorded across all phases. ✅")
        print()


if __name__ == "__main__":
    main()
