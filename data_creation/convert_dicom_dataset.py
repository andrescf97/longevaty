import argparse
import subprocess
import sys
from pathlib import Path

def convert_dataset(input_root: Path, output_root: Path, log_file: Path, limit: int = None):
    """
    Traverse a DICOM dataset organized as input_root/pid/study/series,
    convert each series to .nii.gz using dcm2niix, and log failures.

    - input_root: root folder containing pid subdirectories
    - output_root: root folder to mirror structure and store .nii.gz files
    - log_file: path to write a list of series directories that failed conversion
    - limit: if set, stop after processing this many series (for testing)
    """
    failures = []

    # Gather all series directories
    series_list = []
    for pid_dir in input_root.iterdir():
        if not pid_dir.is_dir():
            continue
        for study_dir in pid_dir.iterdir():
            if not study_dir.is_dir():
                continue
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                series_list.append((pid_dir.name, study_dir.name, series_dir))

    # Apply limit for testing
    if limit is not None and limit > 0:
        series_list = series_list[:limit]
        print(f"🧪 Test mode: processing first {len(series_list)} series")

    # Process series
    for pid, study, series_dir in series_list:
        series_name = series_dir.name
        out_dir = output_root / pid / study / series_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔄 Converting '{series_dir}' → '{out_dir}/{series_name}.nii.gz'")
        cmd = [
            "dcm2niix",
            "-z", "y",               # gzip-compress
            "-f", series_name,         # output filename = series name
            "-o", str(out_dir),        # output directory
            str(series_dir)             # input DICOM folder
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            # Conversion failed; record and continue
            print(f"❌ Conversion failed for {series_dir}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            failures.append(str(series_dir))

    # Log any failures
    if failures:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            for path in failures:
                f.write(path + '\n')
        print(f"⚠️  Logged {len(failures)} failed conversions to '{log_file}'", file=sys.stderr)
    else:
        print("✅ All series converted successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert structured DICOM dataset to .nii.gz with logging"
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory of DICOM dataset (pid/study/series)"
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Root directory for NIfTI output (mirrors input structure)"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("failed_conversions.txt"),
        help="Path to log file listing series that failed conversion"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="If set, only process the first N series (testing mode)"
    )
    args = parser.parse_args()

    convert_dataset(
        args.input_root,
        args.output_root,
        args.log,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
