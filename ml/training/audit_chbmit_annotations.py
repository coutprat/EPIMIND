#!/usr/bin/env python3
"""
Audit CHB-MIT summary files for seizure annotations.

Parses chb01-summary.txt / chb02-summary.txt and reports:
- Which EDF files have seizures
- Seizure start/end times for each
- Total count of seizure intervals
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_summary_file(summary_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse CHB-MIT summary file and extract seizure intervals.
    
    Args:
        summary_path: Path to summary file (e.g., chb01-summary.txt)
        
    Returns:
        Dict mapping EDF filename -> list of (start_sec, end_sec) tuples
    """
    seizures_by_edf = {}
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "File Name: chb01_XX.edf" pattern
        if line.startswith("File Name:"):
            edf_filename = line.replace("File Name:", "").strip()
            seizure_intervals = []
            
            # Read forward to find seizure info for this file
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                
                # Stop at next file
                if next_line.startswith("File Name:"):
                    break
                
                # Parse "Number of Seizures in File: N"
                if "Number of Seizures in File:" in next_line:
                    seizure_count = int(next_line.split(":")[-1].strip())
                    
                    # Read seizure start/end pairs
                    seizures_read = 0
                    k = j + 1
                    while k < len(lines) and seizures_read < seizure_count:
                        seizure_line = lines[k].strip()
                        
                        if "Seizure Start Time:" in seizure_line:
                            # Extract start time: "Seizure Start Time: 2996 seconds"
                            start_match = re.search(r'(\d+)\s+seconds?', seizure_line)
                            if start_match:
                                start_sec = float(start_match.group(1))
                                
                                # Next line should have end time
                                if k + 1 < len(lines):
                                    end_line = lines[k + 1].strip()
                                    if "Seizure End Time:" in end_line:
                                        end_match = re.search(r'(\d+)\s+seconds?', end_line)
                                        if end_match:
                                            end_sec = float(end_match.group(1))
                                            seizure_intervals.append((start_sec, end_sec))
                                            seizures_read += 1
                                            k += 2
                                            continue
                        
                        k += 1
                    
                    # Store results
                    seizures_by_edf[edf_filename] = seizure_intervals
                    break
                
                j += 1
            
            # If no seizures found in this section, still record the file
            if edf_filename not in seizures_by_edf:
                seizures_by_edf[edf_filename] = []
        
        i += 1
    
    return seizures_by_edf


def audit_patient(patient_id: str, data_root: Path) -> None:
    """
    Audit annotations for a single patient.
    
    Args:
        patient_id: Patient ID (e.g., "chb01")
        data_root: Root data directory
    """
    summary_path = data_root / patient_id / f"{patient_id}-summary.txt"
    
    if not summary_path.exists():
        print(f"❌ {patient_id}: Summary file not found at {summary_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"AUDITING: {patient_id}")
    print(f"{'='*70}")
    
    seizures_by_edf = parse_summary_file(summary_path)
    
    # Report findings
    total_edfs = len(seizures_by_edf)
    edfs_with_seizures = [edf for edf, seizures in seizures_by_edf.items() if seizures]
    total_seizure_count = sum(len(seizures) for seizures in seizures_by_edf.values())
    total_seizure_seconds = sum(
        (end - start) for seizures in seizures_by_edf.values() 
        for start, end in seizures
    )
    
    print(f"Total EDF files: {total_edfs}")
    print(f"EDF files with seizures: {len(edfs_with_seizures)}")
    print(f"Total seizure intervals: {total_seizure_count}")
    print(f"Total seizure duration: {total_seizure_seconds} seconds ({total_seizure_seconds/60:.1f} minutes)")
    
    # List each EDF with seizures
    if edfs_with_seizures:
        print(f"\nEDFs with seizures:")
        for edf_filename in sorted(edfs_with_seizures):
            seizures = seizures_by_edf[edf_filename]
            print(f"  {edf_filename}:")
            for start_sec, end_sec in seizures:
                duration = end_sec - start_sec
                print(f"    [{start_sec:>7.1f}s - {end_sec:>7.1f}s] ({duration:>6.1f}s)")
    else:
        print("\n⚠️  No EDF files with seizures found!")
    
    # Summary line for validation
    if total_seizure_count > 0:
        print(f"\n✓ {patient_id}: Found {total_seizure_count} seizure intervals")
        return True
    else:
        print(f"\n✗ {patient_id}: No seizures found!")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Audit CHB-MIT seizure annotations")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/epimind/ml/data/chbmit"),
        help="Root data directory"
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=["chb01", "chb02"],
        help="Patient IDs to audit"
    )
    
    args = parser.parse_args()
    data_root = args.root
    
    print("\n" + "="*70)
    print("CHB-MIT ANNOTATION AUDIT")
    print("="*70)
    
    results = {}
    for patient_id in args.patients:
        found = audit_patient(patient_id, data_root)
        results[patient_id] = found
    
    # Summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    
    total_seizures = sum(
        1 for found in results.values() if found
    )
    
    for patient_id, found in results.items():
        status = "✓ PASSED" if found else "✗ FAILED"
        print(f"{patient_id}: {status}")
    
    # Exit code: 0 if any seizures found, 1 if none
    if total_seizures > 0:
        print(f"\n✓ AUDIT PASSED: Seizure labels are available!")
        exit(0)
    else:
        print(f"\n✗ AUDIT FAILED: No seizures found. Check summary parsing!")
        exit(1)


if __name__ == "__main__":
    main()
