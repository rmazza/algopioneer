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

def get_average_sharpe(pairs):
    """Extracts and averages Sharpe ratios if available in the config."""
    sharpes = [float(p.get('sharpe_ratio', 0)) for p in pairs if 'sharpe_ratio' in p]
    if not sharpes:
        return 0.0
    return sum(sharpes) / len(sharpes)

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

    # 1. Check for Symbol Changes (Mandatory Update)
    if current_sig != new_sig:
        print(f"Pairs changed! Old: {len(current_sig)}, New: {len(new_sig)}")
        added = new_sig - current_sig
        removed = current_sig - new_sig
        if added:
            print(f"  + Added: {added}")
        if removed:
            print(f"  - Removed: {removed}")
        sys.exit(1)
    
    # 2. Check for Statistical Improvement (Threshold Update)
    # If symbols are identical, we only update if the new parameters offer 
    # a significant improvement (e.g. 10% better Sharpe).
    IMPROVEMENT_THRESHOLD = 1.10 
    
    current_sharpe = get_average_sharpe(current_pairs)
    new_sharpe = get_average_sharpe(new_pairs)
    
    if current_sharpe > 0 and (new_sharpe / current_sharpe) >= IMPROVEMENT_THRESHOLD:
        print(f"Significant improvement detected! Sharpe: {current_sharpe:.2f} -> {new_sharpe:.2f}")
        sys.exit(1)

    print(f"No significant changes or improvement ({current_sharpe:.2f} vs {new_sharpe:.2f}). Stability preferred.")
    sys.exit(0)

if __name__ == "__main__":
    main()
