#destructive_reviewer.py
import json
import os

def review_destructive_data(raw_data_file="destructive_data_raw.jsonl", reviewed_data_file="destructive_data_reviewed.jsonl"):
    """Simulates human review of potentially destructive data."""

    if not os.path.exists(raw_data_file):
        print(f"Error: Raw data file '{raw_data_file}' not found.")
        return

    reviewed_data = []
    with open(raw_data_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            print("-" * 20)
            print("Reviewing Item:")
            print(f"  Question: {item['messages'][1]['content']}")
            print(f"  Answer: {item['messages'][2]['content']}")
            print(f"  Source: {item['source']}")

            while True:
                action = input("Action (approve/reject/modify/quit): ").lower()
                if action in ("approve", "reject", "modify", "quit"):
                    break
                print("Invalid action.  Please enter 'approve', 'reject', 'modify', or 'quit'.")
            if action == "quit":
                break;

            if action == "approve":
                reviewed_data.append(item)
            elif action == "modify":
                modified_answer = input("Enter the modified (safe) answer: ")
                item['messages'][2]['content'] = modified_answer
                item['reason'] = item.get('reason', '') + ' (Modified by human review)'
                reviewed_data.append(item)
            elif action == 'reject':
                print("Item Rejected")
            # "reject" - item is simply skipped

    with open(reviewed_data_file, 'w') as f:
        for item in reviewed_data:
            f.write(json.dumps(item) + "\n")
    print(f"Reviewed data saved to '{reviewed_data_file}'")

if __name__ == "__main__":
    review_destructive_data()