#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nbclient",
#   "nbformat",
#   "ipykernel",
# ]
# ///
"""Execute a Jupyter notebook in-place, capturing outputs into the .ipynb file.

Reads the notebook's PEP 723 dependencies and bootstraps the environment
automatically via `uv run`, so you don't need a pre-built environment.

Usage:
    # Execute a single notebook (reads its PEP 723 deps automatically):
    uv run _scripts/execute_notebook.py posts/probability/index.ipynb

    # Execute with extra timeout per cell (default 600s):
    uv run _scripts/execute_notebook.py --timeout 1200 posts/switchpoint/index.ipynb

    # Execute all modified notebooks:
    for nb in posts/*/index.ipynb; do
        uv run _scripts/execute_notebook.py "$nb"
    done

    # Continue on cell errors (capture error output instead of stopping):
    uv run _scripts/execute_notebook.py --allow-errors posts/foo/index.ipynb

How it works:
    Phase 1: Script reads the notebook, extracts PEP 723 dependencies.
    Phase 2: If the notebook has extra deps not in our environment, re-invokes
             itself via `uv run --with <deps>` so the kernel has everything.
    Phase 3: Uses nbclient to execute the notebook and write outputs in-place.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def extract_pep723_deps(nb_path: str) -> list[str]:
    """Extract PEP 723 dependencies from a notebook's metadata cell."""
    with open(nb_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "# /// script" not in src:
            continue
        # Parse the dependencies list
        deps = []
        in_deps = False
        for line in src.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                line = line[2:]
            elif line.startswith("#"):
                line = line[1:]
            else:
                continue
            line = line.strip()
            if line == "dependencies = [":
                in_deps = True
                continue
            if in_deps:
                if line == "]":
                    break
                # Strip quotes and trailing comma
                dep = line.strip().strip(",").strip('"').strip("'")
                if dep:
                    deps.append(dep)
        return deps
    return []


def check_deps_available(deps: list[str]) -> list[str]:
    """Return list of deps that are NOT importable in current environment."""
    missing = []
    # Map package names to import names
    import_map = {
        "scikit-learn": "sklearn",
        "ipython": "IPython",
        "pymc": "pymc",
        "pytensor": "pytensor",
    }
    for dep in deps:
        # Strip version specifiers
        pkg = re.split(r"[<>=!;]", dep)[0].strip()
        import_name = import_map.get(pkg, pkg.replace("-", "_"))
        try:
            __import__(import_name)
        except ImportError:
            missing.append(dep)
    return missing


def relaunch_with_deps(nb_path: str, extra_deps: list[str], args: argparse.Namespace):
    """Re-invoke this script under `uv run --with <deps>`."""
    cmd = ["uv", "run", "--with", "nbclient", "--with", "nbformat", "--with", "ipykernel"]
    for dep in extra_deps:
        cmd.extend(["--with", dep])
    # Use "python" (not sys.executable) so uv provides the interpreter
    # from the new environment with all --with deps installed
    cmd.extend(["python", os.path.abspath(__file__), "--_bootstrapped"])
    if args.timeout != 600:
        cmd.extend(["--timeout", str(args.timeout)])
    if args.allow_errors:
        cmd.append("--allow-errors")
    cmd.append(nb_path)

    print(f"  Re-launching with deps: {', '.join(extra_deps)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def execute_notebook(nb_path: str, timeout: int = 600, allow_errors: bool = False):
    """Execute the notebook in-place using nbclient."""
    import nbformat
    from nbclient import NotebookClient

    print(f"  Executing {nb_path} ...")

    nb = nbformat.read(nb_path, as_version=4)

    # Set up the client
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=allow_errors,
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(nb_path)) or "."}},
    )

    # Execute
    client.execute()

    # Write back
    nbformat.write(nb, nb_path)
    print(f"  Done: {nb_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute a notebook in-place with output capture.")
    parser.add_argument("notebook", help="Path to the .ipynb file")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per cell in seconds (default: 600)")
    parser.add_argument("--allow-errors", action="store_true", help="Continue execution on cell errors")
    parser.add_argument("--_bootstrapped", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    nb_path = args.notebook
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} not found", file=sys.stderr)
        sys.exit(1)

    if not args._bootstrapped:
        # Phase 1: Check if we need to re-launch with notebook's deps
        nb_deps = extract_pep723_deps(nb_path)
        if nb_deps:
            missing = check_deps_available(nb_deps)
            if missing:
                relaunch_with_deps(nb_path, nb_deps, args)
                return  # relaunch_with_deps calls sys.exit

    # Phase 2: We have all deps, execute the notebook
    execute_notebook(nb_path, timeout=args.timeout, allow_errors=args.allow_errors)


if __name__ == "__main__":
    main()
