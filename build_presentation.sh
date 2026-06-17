#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is not installed. Install a TeX Live distribution with XeLaTeX support before building." >&2
  echo "On Ubuntu this is typically: sudo apt install latexmk texlive-xetex texlive-latex-extra texlive-fonts-recommended" >&2
  exit 1
fi

latexmk -xelatex -interaction=nonstopmode -synctex=1 fyp_presentation.tex
