import json
import argparse
import os

def merge_datasets(training_file, reviewed_destructive_file, output_file):
    """Merges (or just copies) training data and reviewed destructive data."""

    if not os.path.exists(training_file):
        print(f"Error: Training file '{training_file}' not found.")
        return

    merged_data = []

    # Always include the training data
    with open(training_file, 'r') as f:
        for line in f:
            merged_data.append(json.loads(line))

    with open(output_file, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + "\n")

    print(f"Merged data saved to '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Merge training data with reviewed destructive data (optional).")
    parser.add_argument("training_file", help="Path to the raw training data file (JSONL).")
    parser.add_argument("reviewed_destructive_file", help="Path to the reviewed destructive data file (JSONL).")
    parser.add_argument("output_file", help="Path to the output file for the merged data (JSONL).")
    parser.add_argument("--merge", action="store_true", help="Merge the destructive data into the training data.  If not set, only the training data will be used.")

    args = parser.parse_args()
    merge_datasets(args.training_file, args.reviewed_destructive_file, args.output_file, args.merge)

if __name__ == "__main__":
    main()