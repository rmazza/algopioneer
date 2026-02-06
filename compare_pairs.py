import sys
import json
import os

def load_pairs(filepath):
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return []

def get_pair_signature(pairs):
    """Returns a set of (spot, future) tuples to identify the pair universe."""
    return set((p['spot_symbol'], p['future_symbol']) for p in pairs)

def main():
    if len(sys.argv) < 3:
        print("Usage: compare_pairs.py <current_json> <new_json>")
        sys.exit(0)

    current_file = sys.argv[1]
    new_file = sys.argv[2]

    current_pairs = load_pairs(current_file)
    new_pairs = load_pairs(new_file)

    if not new_pairs:
        print("New pairs file is empty or invalid. Keeping current.")
        sys.exit(0)

    if not current_pairs:
        print("No current pairs found. Deploying new.")
        sys.exit(1)

    current_sig = get_pair_signature(current_pairs)
    new_sig = get_pair_signature(new_pairs)

    # Logic: If the set of pairs has changed, update.
    # We could look at parameters too, but symbol changes are the most important.
    if current_sig != new_sig:
        print(f"Pairs changed! Old: {len(current_sig)}, New: {len(new_sig)}")
        added = new_sig - current_sig
        removed = current_sig - new_sig
        if added:
            print(f"  + Added: {added}")
        if removed:
            print(f"  - Removed: {removed}")
        sys.exit(1)
    
    # If sets are same, check for parameter drift? 
    # For now, stability is preferred. If symbols are same, stick with current state 
    # to avoid restarting the bot unnecessarily.
    print("Pair set is identical. No changes needed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
