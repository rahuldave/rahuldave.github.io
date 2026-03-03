#!/bin/bash
# Build JupyterLite static site into _site/lab/
set -e

echo "Building JupyterLite..."

# Find the labextensions path where jupyterlite-pyodide-kernel is installed.
# Homebrew puts it under /opt/homebrew/share/jupyter/labextensions/ which may
# not be in the default Jupyter data path, so we pass it explicitly.
LABEXT_PATH=$(python3 -c "
import importlib.resources, pathlib
pkg = importlib.import_module('jupyterlite_pyodide_kernel')
# The labextension lives alongside the package's share/jupyter/labextensions
site_pkg = pathlib.Path(pkg.__file__).parent.parent
candidates = [
    site_pkg / 'share' / 'jupyter' / 'labextensions',
    pathlib.Path('/opt/homebrew/share/jupyter/labextensions'),
]
for c in candidates:
    if (c / '@jupyterlite' / 'pyodide-kernel-extension').is_dir():
        print(c)
        break
" 2>/dev/null)

if [ -n "$LABEXT_PATH" ]; then
    echo "Using labextensions from: $LABEXT_PATH"
    jupyter lite build --lite-dir _lab --output-dir _site/lab \
        --FederatedExtensionAddon.extra_labextensions_path="$LABEXT_PATH"
else
    echo "Warning: could not find pyodide-kernel-extension, building without it"
    jupyter lite build --lite-dir _lab --output-dir _site/lab
fi

echo "JupyterLite built to _site/lab/"
