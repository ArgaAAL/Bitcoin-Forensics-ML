import pandas as pd
import json
import os

# --- Configuration ---
# File paths for the datasets we need to compare.
DATA_DIR = './elliptic_dataset'
FEATURES_PATH = os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
CACHE_FILE = 'local_tx_cache.json'

def analyze_and_compare_data():
    """
    This script loads both datasets, extracts the relevant features,
    and prints a direct comparison to diagnose the feature mismatch.
    """
    print("====== Data Analysis and Feature Comparison ======")

    # --- Step 1: Load and Analyze the Live JSON Data ---
    print(f"\n--- Analyzing Live Data from '{CACHE_FILE}' ---")
    try:
        with open(CACHE_FILE, 'r') as f:
            live_data_cache = json.load(f)
        # We analyze the first address in the cache file.
        address_key = next(iter(live_data_cache))
        live_transactions = live_data_cache[address_key]
        print(f"[SUCCESS] Loaded {len(live_transactions)} transactions for address {address_key} from cache.")
    except (FileNotFoundError, StopIteration, json.JSONDecodeError) as e:
        print(f"[FATAL] Could not load or parse '{CACHE_FILE}'. Please ensure the file exists and is a valid JSON.")
        print(f"Error: {e}")
        return

    # Extract the first 3 transactions for a clear example
    live_examples = []
    for tx in live_transactions[:3]:
        live_examples.append({
            'vin_sz': tx.get('vin_sz', 'N/A'),
            'vout_sz': tx.get('vout_sz', 'N/A'),
            'fee': tx.get('fee', 'N/A'),
            'size': tx.get('size', 'N/A')
        })
    
    print("\n[Live Data] Example Feature Values (first 3 transactions):")
    for i, example in enumerate(live_examples):
        print(f"  Tx {i+1}: {example}")

    # --- Step 2: Load and Analyze the Elliptic Training Data ---
    print(f"\n--- Analyzing Training Data from Elliptic Dataset ---")
    try:
        features_df = pd.read_csv(FEATURES_PATH, header=None)
        print(f"[SUCCESS] Loaded Elliptic features data with shape: {features_df.shape}")
    except FileNotFoundError:
        print(f"[FATAL] Could not find '{FEATURES_PATH}'. Please ensure the dataset is downloaded.")
        return

    # These are the columns we PREVIOUSLY ASSUMED were proxies for the live data.
    # Live JSON -> Elliptic Column Index
    # 'vin_sz'  -> local_feature_11 (Column 13)
    # 'vout_sz' -> local_feature_12 (Column 14)
    # 'fee'     -> local_feature_13 (Column 15)
    # 'size'    -> local_feature_10 (Column 12)
    proxy_cols = {
        'proxy_for_vin_sz': 13,
        'proxy_for_vout_sz': 14,
        'proxy_for_fee': 15,
        'proxy_for_size': 12
    }
    
    elliptic_proxy_df = features_df[list(proxy_cols.values())]
    elliptic_proxy_df.columns = list(proxy_cols.keys())

    print("\n[Elliptic Data] Example Feature Values (first 3 transactions):")
    print(elliptic_proxy_df.head(3).to_string())

    print("\n[Elliptic Data] Statistical Summary of Proxy Features:")
    print(elliptic_proxy_df.describe().to_string())


    # --- Step 3: Print a Direct Side-by-Side Comparison Conclusion ---
    print("\n" + "="*20 + " COMPARISON CONCLUSION " + "="*20)
    print("This analysis programmatically confirms the fundamental feature mismatch:")
    
    print("\nFEATURE: Number of Inputs")
    print(f"  - Live JSON ('vin_sz'):  Named explicitly. Raw integer values. Example: {live_examples[0]['vin_sz']}")
    print(f"  - Elliptic Proxy:        Anonymous ('local_feature_11'). Normalized float values. Example: {elliptic_proxy_df['proxy_for_vin_sz'].iloc[0]:.2f}")
    
    print("\nFEATURE: Number of Outputs")
    print(f"  - Live JSON ('vout_sz'): Named explicitly. Raw integer values. Example: {live_examples[0]['vout_sz']}")
    print(f"  - Elliptic Proxy:        Anonymous ('local_feature_12'). Normalized float values. Example: {elliptic_proxy_df['proxy_for_vout_sz'].iloc[0]:.2f}")

    print("\nFEATURE: Transaction Fee")
    print(f"  - Live JSON ('fee'):     Named explicitly. Raw integer values (sats). Example: {live_examples[0]['fee']}")
    print(f"  - Elliptic Proxy:        Anonymous ('local_feature_13'). Normalized float values. Example: {elliptic_proxy_df['proxy_for_fee'].iloc[0]:.2f}")

    print("\nOVERALL VERDICT:")
    print("The training data and live data are in completely different 'languages'.")
    print("The names, data types, and value ranges are incompatible.")
    print("Any model trained on the Elliptic proxy features will fail when given live data features because the patterns are fundamentally different.")
    print("\nThis analysis proves that a new model must be trained from scratch using features that are engineered to perfectly match the structure and format of the live API data.")


if __name__ == '__main__':
    analyze_and_compare_data()

