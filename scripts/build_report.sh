#!/bin/bash
# Rebuild everything the report derives from the result files, then the PDF.
#   scripts/build_report.sh           final build: stops if anything is missing or inconsistent
#   scripts/build_report.sh --draft   same steps, but only lists what is missing
# Nothing from an earlier build is reused: the generated figures, tables and numbers
# are deleted first. Only the runs named in results/selected_runs.json are read.
set -euo pipefail
cd "$(dirname "$0")/.."
strict="--strict"; [[ "${1:-}" == "--draft" ]] && strict=""
if [[ -n "${ATTN_PREVIEW:-}" ]]; then
    echo "ATTN_PREVIEW is set: a report is built from the explicit selection of finished runs only" >&2; exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-results/matplotlib-cache}"
rm -rf report/figures report/generated
mkdir -p report/figures report/generated output/pdf
python3 scripts/audit_results.py > /dev/null     # the rules, before anything is computed from the records
python3 scripts/figures_concept.py
python3 scripts/make_figures.py $strict
python3 scripts/make_tables.py
python3 scripts/audit_results.py                 # again: now it also records which output used which file
python3 scripts/make_numbers.py $strict
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf report/report.tex > output/pdf/build.log 2>&1 \
    || { grep -n -A 6 "^!" output/pdf/report.log | head -40; exit 1; }
# The gate: no number without data (printed as ??key??), no figure or table that was
# not generated, no undefined reference, no text outside the page.
python3 - "$strict" <<'PY'
import re, sys
import pypdf
strict = sys.argv[1] == "--strict"
reader = pypdf.PdfReader("output/pdf/report.pdf")
text = [page.extract_text() or "" for page in reader.pages]
problems = []
missing = {}
for number, page in enumerate(text, 1):
    for key in re.findall(r"\?\?([A-Za-z0-9._]+)\?\?", page):
        missing.setdefault(key, number)
    if "not generated yet" in page:
        problems.append(f"page {number}: a figure or table was not generated")
problems += [f"page {number}: number without data: {key}" for key, number in sorted(missing.items())]
log = open("output/pdf/report.log", errors="replace").read()
undefined = sorted(set(re.findall(r"LaTeX Warning: (?:Reference|Citation) `([^']+)' on page", log)))
problems += [f"undefined reference: {name}" for name in undefined]
overfull = re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)
problems += [f"text {width} pt outside the margin" for width in overfull if float(width) > 2]
print(f"output/pdf/report.pdf: {len(reader.pages)} pages; numbers without data: {len(missing)}; other problems: {len(problems) - len(missing)}")
for problem in problems:
    print("  " + problem)
if problems and strict:
    raise SystemExit("the report is not complete; use --draft to build it anyway")
PY
