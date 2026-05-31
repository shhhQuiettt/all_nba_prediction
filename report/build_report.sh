#!/bin/bash

set -e 

INPUT="report/report.md"
HTML_OUT="report/report.html"
PDF_OUT="report/report.pdf"
CSS_FILE="report/style.css"


if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found!"
    exit 1
fi

echo "📄 Generating HTML..."
pandoc "$INPUT" \
    --from markdown \
    --standalone \
    --embed-resources \
    --css="$CSS_FILE" \
    -o "$HTML_OUT"


echo "📕 Generating PDF..."
pandoc "$INPUT" \
     -V geometry:margin=1in \
     -V mainfont="Helvetica" \
     -V sansfont="Helvetica" \
     -V monofont="Courier" \
     -V fontsize=12pt \
    --highlight-style=tango \
    --css="$CSS_FILE"  \
    -o "$PDF_OUT"

echo "Build complete! Check $HTML_OUT and $PDF_OUT."
