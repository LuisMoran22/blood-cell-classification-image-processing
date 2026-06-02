"""Validate that the dataset2-master structure is readable."""
from __future__ import annotations
import argparse
from pprint import pprint
from dataset_utils import summarize_dataset

def main():
    parser = argparse.ArgumentParser(description="Check dataset structure and image counts.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root or images folder.")
    args = parser.parse_args()
    summary = summarize_dataset(args.data_dir)
    pprint(summary)
    total_train = sum(summary["splits"].get("TRAIN", {}).values())
    total_test = sum(summary["splits"].get("TEST", {}).values())
    if total_train == 0 or total_test == 0:
        raise ValueError("TRAIN or TEST split appears to be empty. Check the dataset path.")
    print(f"Dataset ready. TRAIN images: {total_train} | TEST images: {total_test}")
if __name__ == "__main__": main()
