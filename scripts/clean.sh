#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

DRY_RUN=false
CLEAN_OUTPUTS=false
CLEAN_ALL=false

usage() {
  cat <<'USAGE'
Usage: scripts/clean.sh [options]

Removes generated artifacts to keep the repo tidy.
By default, only cleans generated files in the repo root and coverage outputs.

Options:
  --dry-run       Show what would be removed without deleting
  --outputs       Also clean outputs/ generated artifacts
  --all           Clean everything (root + outputs)
  -h, --help      Show this help

Examples:
  scripts/clean.sh --dry-run
  scripts/clean.sh --outputs
  scripts/clean.sh --all
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --outputs) CLEAN_OUTPUTS=true ;;
    --all) CLEAN_ALL=true ; CLEAN_OUTPUTS=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

ROOT_FILES=(
  demo_llm_trace.json
  demo_real_trace.json
  demo_trace.json
  llm_report.md
  llm_report.html
  real_report.md
  real_report.html
  demo_report.md
  demo_report.html
  scaling_summary.json
  scaling_summary.csv
)
ROOT_GLOBS=(
  'trace_*.json'
  'report_*.md'
  'report_*.html'
)
COVERAGE_DIRS=(
  coverage
  coverage-html
)
OUTPUTS_DIR="$ROOT_DIR/outputs"

collect_matches() {
  local -n out=$1
  # Root files
  for f in "${ROOT_FILES[@]}"; do
    if [[ -e "$ROOT_DIR/$f" ]]; then
      out+=("$ROOT_DIR/$f")
    fi
  done
  # Root globs
  for g in "${ROOT_GLOBS[@]}"; do
    for f in $ROOT_DIR/$g; do
      [[ -e "$f" ]] && out+=("$f")
    done
  done
  # Coverage dirs
  for d in "${COVERAGE_DIRS[@]}"; do
    if [[ -d "$ROOT_DIR/$d" ]]; then
      out+=("$ROOT_DIR/$d/")
    fi
  done
  # Outputs if requested
  if [[ "$CLEAN_OUTPUTS" == true && -d "$OUTPUTS_DIR" ]]; then
    out+=("$OUTPUTS_DIR/")
  fi
}

matches=()
collect_matches matches

if [[ "$DRY_RUN" == true ]]; then
  echo "[DRY RUN] Would remove the following paths:";
  for m in "${matches[@]}"; do
    echo "  $m"
  done
  exit 0
fi

# Perform deletions
for m in "${matches[@]}"; do
  if [[ -d "$m" ]]; then
    echo "Removing directory: $m"
    rm -rf "$m"
  else
    echo "Removing file: $m"
    rm -f "$m"
  fi
done

echo "Cleanup complete."
