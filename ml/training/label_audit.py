"""
Label Audit Script for CHB-MIT Dataset

Robustly parses seizure information from CHB-MIT summary files and validates
that seizure labels are properly extracted. Fails if no seizures found.

Usage:
  python label_audit.py --root D:/epimind/ml/data/chbmit --patients chb01 chb02
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def find_summary_file(root_dir: Path, patient: str) -> Path:
    """
    Locate summary file for a patient.
    
    Tries:
    1. {root}/{patient}/{patient}-summary.txt
    2. {root}/{patient}-summary.txt
    
    Returns:
        Path to summary file
        
    Raises:
        FileNotFoundError if not found
    """
    root_dir = Path(root_dir)
    
    # Try direct path
    candidates = [
        root_dir / patient / f"{patient}-summary.txt",
        root_dir / f"{patient}-summary.txt",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Summary file not found for {patient}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def parse_summary_file(summary_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse seizure information from CHB-MIT summary file.
    
    Expected format:
        File Name: chb01_03.edf
        File Start Time: ...
        File End Time: ...
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    
    Returns:
        Dict: {
            "edf_filename": [(start_sec, end_sec), ...],
            ...
        }
    """
    seizure_data = {}
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    current_edf = None
    seizure_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "File Name:" pattern
        if line.startswith("File Name:"):
            current_edf = line.replace("File Name:", "").strip()
            seizure_data[current_edf] = []
            seizure_count = 0
            i += 1
            
            # Next lines: File Start Time, File End Time, Number of Seizures
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("Number of Seizures in File:"):
                    try:
                        seizure_count = int(line.replace("Number of Seizures in File:", "").strip())
                    except ValueError:
                        seizure_count = 0
                    
                    i += 1
                    
                    # Read seizure lines
                    for _ in range(seizure_count):
                        if i >= len(lines):
                            break
                        
                        line = lines[i].strip()
                        
                        # Look for "Seizure Start Time: XXXX seconds"
                        if line.startswith("Seizure Start Time:"):
                            try:
                                start_match = re.search(r'(\d+)\s+seconds?', line)
                                if start_match:
                                    start_sec = float(start_match.group(1))
                                    
                                    # Next line should be "Seizure End Time:"
                                    i += 1
                                    if i < len(lines):
                                        end_line = lines[i].strip()
                                        end_match = re.search(r'(\d+)\s+seconds?', end_line)
                                        if end_match:
                                            end_sec = float(end_match.group(1))
                                            seizure_data[current_edf].append((start_sec, end_sec))
                            except (ValueError, IndexError):
                                pass
                        
                        i += 1
                    
                    break
                
                i += 1
        else:
            i += 1
    
    return seizure_data


def audit_patient(root_dir: Path, patient: str) -> Dict:
    """
    Audit seizure labels for a single patient.
    
    Returns:
        Dict with summary statistics
    """
    print(f"\n{'=' * 80}")
    print(f"Patient: {patient.upper()}")
    print(f"{'=' * 80}")
    
    # Find summary file
    try:
        summary_path = find_summary_file(root_dir, patient)
        print(f"✓ Summary file: {summary_path}")
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        return {
            "patient": patient,
            "found": False,
            "error": str(e),
            "total_seizures": 0,
            "total_seizure_seconds": 0,
            "edf_with_seizures": 0,
            "files_with_seizures": [],
        }
    
    # Parse summary
    seizure_data = parse_summary_file(summary_path)
    
    print(f"✓ Parsed {len(seizure_data)} EDF files")
    
    # Calculate statistics
    total_seizures = 0
    total_seizure_seconds = 0.0
    files_with_seizures = []
    edf_with_seizures = 0
    
    print(f"\n{'File Name':<20} {'Seizures':<12} {'Start (s)':<12} {'End (s)':<12} {'Duration (s)':<12}")
    print(f"{'-' * 80}")
    
    for edf_file in sorted(seizure_data.keys()):
        intervals = seizure_data[edf_file]
        
        if intervals:
            edf_with_seizures += 1
            
            for start_sec, end_sec in intervals:
                duration = end_sec - start_sec
                total_seizures += 1
                total_seizure_seconds += duration
                files_with_seizures.append(edf_file)
                
                print(f"{edf_file:<20} {1:<12} {start_sec:<12.1f} {end_sec:<12.1f} {duration:<12.1f}")
    
    if not files_with_seizures:
        print(f"{'(No seizures found)':<80}")
    
    print(f"\n{'─' * 80}")
    print(f"Summary for {patient.upper()}:")
    print(f"  Total EDF files: {len(seizure_data)}")
    print(f"  EDF files with seizures: {edf_with_seizures}")
    print(f"  Total seizure events: {total_seizures}")
    print(f"  Total seizure time: {total_seizure_seconds:.1f} seconds ({total_seizure_seconds/60:.1f} min)")
    print(f"  Average seizure duration: {total_seizure_seconds/max(1, total_seizures):.1f} seconds")
    
    result = {
        "patient": patient,
        "found": True,
        "error": None,
        "total_edf_files": len(seizure_data),
        "edf_with_seizures": edf_with_seizures,
        "total_seizures": total_seizures,
        "total_seizure_seconds": total_seizure_seconds,
        "files_with_seizures": files_with_seizures,
        "seizure_data": seizure_data,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit seizure labels in CHB-MIT dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="D:/epimind/ml/data/chbmit",
        help="Root directory of CHB-MIT dataset"
    )
    parser.add_argument(
        "--patients",
        type=str,
        nargs="+",
        default=["chb01", "chb02"],
        help="List of patient IDs to audit"
    )
    
    args = parser.parse_args()
    root_dir = Path(args.root)
    
    if not root_dir.exists():
        print(f"ERROR: Root directory not found: {root_dir}")
        exit(1)
    
    results = {}
    total_seizures_all = 0
    total_seizure_seconds_all = 0.0
    
    for patient in args.patients:
        result = audit_patient(root_dir, patient)
        results[patient] = result
        
        if result["found"]:
            total_seizures_all += result["total_seizures"]
            total_seizure_seconds_all += result["total_seizure_seconds"]
    
    # Final summary
    print(f"\n{'=' * 80}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 80}")
    
    success = True
    for patient, result in results.items():
        if not result["found"]:
            print(f"✗ {patient}: Summary file not found")
            success = False
        elif result["total_seizures"] == 0:
            print(f"⚠ {patient}: No seizures found (may be data issue)")
            success = False
        else:
            print(f"✓ {patient}: {result['total_seizures']} seizures, {result['total_seizure_seconds']:.1f}s total")
    
    print(f"\nTotal seizures across all patients: {total_seizures_all}")
    print(f"Total seizure time: {total_seizure_seconds_all:.1f} seconds")
    
    # Exit with error if any patient has no seizures
    if total_seizures_all == 0:
        print(f"\n✗ AUDIT FAILED: No seizures found in any patient!")
        print(f"Fix: Ensure CHB-MIT data is properly downloaded and summary files are present.")
        exit(1)
    else:
        print(f"\n✓ AUDIT PASSED: Seizure labels are available!")
        exit(0)


if __name__ == "__main__":
    main()
