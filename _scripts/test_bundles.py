#!/usr/bin/env python3
"""Test generated notebook bundles by running them via `juv exec`.

Unzips each bundle to a temp directory, runs `uvx juv exec index.ipynb`,
captures results, and writes a test report.

Usage:
    python3 _scripts/test_bundles.py [--site-dir _site] [--report _site/test-report.json] [--timeout 300]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import zipfile


def test_bundle(zip_path: Path, timeout: int = 300) -> dict:
    """Test a single bundle. Returns result dict."""
    slug = zip_path.stem
    result = {
        "slug": slug,
        "zip_path": str(zip_path),
        "status": "unknown",
        "exit_code": None,
        "duration_s": None,
        "stderr_snippet": "",
    }

    # Create temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"bundle-test-{slug}-"))

    try:
        # Unzip
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)

        nb_path = tmp_dir / "index.ipynb"
        if not nb_path.exists():
            result["status"] = "error"
            result["stderr_snippet"] = "index.ipynb not found in zip"
            return result

        # Run via juv exec
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["uvx", "juv", "exec", "index.ipynb"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            result["exit_code"] = proc.returncode
            result["duration_s"] = round(duration, 1)
            result["status"] = "pass" if proc.returncode == 0 else "fail"

            # Capture last 500 chars of stderr for diagnostics
            if proc.stderr:
                result["stderr_snippet"] = proc.stderr[-500:]
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            result["status"] = "timeout"
            result["duration_s"] = round(duration, 1)
            result["stderr_snippet"] = f"Timed out after {timeout}s"
        except FileNotFoundError:
            result["status"] = "error"
            result["stderr_snippet"] = "uvx or juv not found. Install with: pip install uv"

    except Exception as e:
        result["status"] = "error"
        result["stderr_snippet"] = str(e)[:500]
    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test notebook bundles via juv exec")
    parser.add_argument("--site-dir", default="_site", help="Site output directory")
    parser.add_argument("--report", default=None, help="Output report JSON path")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per notebook (seconds)")
    parser.add_argument("--slug", default=None, help="Test only this slug (for debugging)")
    args = parser.parse_args()

    site_dir = Path(args.site_dir)

    if args.report is None:
        args.report = str(site_dir / "test-report.json")

    # Find all zip bundles
    zips = sorted(site_dir.glob("posts/*/*.zip"))
    if args.slug:
        zips = [z for z in zips if z.stem == args.slug]

    if not zips:
        print("No zip bundles found. Run `make bundles` first.")
        sys.exit(1)

    print(f"Testing {len(zips)} bundles (timeout: {args.timeout}s each)")
    print("=" * 60)

    results = []
    counts = {"pass": 0, "fail": 0, "error": 0, "timeout": 0}

    for zip_path in zips:
        slug = zip_path.stem
        print(f"  {slug}...", end=" ", flush=True)
        result = test_bundle(zip_path, timeout=args.timeout)
        results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1

        status_icon = {
            "pass": "PASS",
            "fail": "FAIL",
            "error": "ERR",
            "timeout": "TIME",
        }.get(result["status"], "???")

        duration = f"({result['duration_s']}s)" if result["duration_s"] else ""
        print(f"{status_icon} {duration}")

        if result["status"] != "pass" and result["stderr_snippet"]:
            # Print first line of error
            first_line = result["stderr_snippet"].strip().split("\n")[-1]
            print(f"    -> {first_line[:100]}")

    # Write report
    report = {
        "summary": counts,
        "total": len(results),
        "results": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Results: {counts.get('pass', 0)} pass, {counts.get('fail', 0)} fail, "
          f"{counts.get('error', 0)} error, {counts.get('timeout', 0)} timeout")
    print(f"Report written to {report_path}")

    # Exit with non-zero if any failures (but don't block CI for now)
    if counts.get("fail", 0) > 0 or counts.get("error", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
