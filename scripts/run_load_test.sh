#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# run_load_test.sh — 3-phase load test: warmup → ramp → stress
#
# Usage:
#   ./scripts/run_load_test.sh                     # Default: localhost:8000
#   ./scripts/run_load_test.sh http://alb-dns:80   # Custom host
#
# Prerequisites:
#   pip install locust
#   Server running at target host
# ─────────────────────────────────────────────────────────────────────

HOST="${1:-http://localhost:8000}"
RESULTS="tests/load/results"
mkdir -p "$RESULTS"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Load Test Target: $HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "=== Phase 1: Warmup (5 users, 30s) ==="
locust -f tests/load/locustfile.py --host "$HOST" --headless \
  -u 5 -r 2 -t 30s \
  --csv "$RESULTS/warmup" --html "$RESULTS/warmup.html" \
  2>&1 | tail -5

echo ""
echo "=== Phase 2: Ramp (50 users, 2min) ==="
locust -f tests/load/locustfile.py --host "$HOST" --headless \
  -u 50 -r 10 -t 2m \
  --csv "$RESULTS/ramp" --html "$RESULTS/ramp.html" \
  2>&1 | tail -5

echo ""
echo "=== Phase 3: Stress (200 users, 1min) ==="
locust -f tests/load/locustfile.py --host "$HOST" --headless \
  -u 200 -r 50 -t 1m \
  --csv "$RESULTS/stress" --html "$RESULTS/stress.html" \
  2>&1 | tail -5

echo ""
echo "=== Generating report ==="
python scripts/generate_metrics_report.py "$RESULTS"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results saved to: $RESULTS/"
echo "  HTML reports: warmup.html, ramp.html, stress.html"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
