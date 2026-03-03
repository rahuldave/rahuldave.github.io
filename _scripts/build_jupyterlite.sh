#!/bin/bash
# Build JupyterLite static site into _site/lab/
set -e

echo "Building JupyterLite..."
jupyter lite build --lite-dir _lab --output-dir _site/lab
echo "JupyterLite built to _site/lab/"
