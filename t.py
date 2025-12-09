import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
INPUT_FILE = 'bitmex_incremental_book_L2.csv'
OUTPUT_FILE = 'bitmex_incremental_book_L2_5M_sample.csv.gz'
SAMPLE_SIZE = 5_000_000 # 5 million rows for benchmarking
RANDOM_SEED = 42 # For reproducible sampling
# ---------------------

def downcast_df(df):
    """Reduces the memory usage of a DataFrame by downcasting numeric types."""
    print("Applying memory downcasting (64-bit to 32-bit conversion)...")
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    # Downcast integers
    for col in df.select_dtypes(include=['int64']).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
             df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def process_and_save_data():
    """Loads the full compressed data, samples, downcasts, and saves."""
    try:
        print(f"1. Reading full data from {INPUT_FILE} as GZIP...")
        
        # Load the entire DataFrame. This step requires enough memory on your local machine.
        df_full = pd.read_csv(INPUT_FILE, 
                              compression='gzip', 
                              encoding='utf-8', 
                              sep=',') 

        print(f"   -> Full data loaded. Total rows: {len(df_full):,}")

    except Exception as e:
        print("\nCRITICAL ERROR during file loading:")
        print("This script failed because your original CSV is either corrupt or the compression/delimiter is wrong.")
        print(f"Details: {e}")
        print("Please ensure the last successful loading parameters (compression='gzip', sep=',') are correct.")
        sys.exit(1)

    if len(df_full) < SAMPLE_SIZE:
        print(f"Warning: File has fewer rows ({len(df_full):,}) than requested sample size. Using all rows.")
        df_sampled = df_full
    else:
        print(f"2. Randomly sampling {SAMPLE_SIZE:,} rows...")
        # Randomly sample the desired number of rows
        df_sampled = df_full.sample(n=SAMPLE_SIZE, replace=False, random_state=RANDOM_SEED)

    # 3. Apply Downcasting
    df_optimized = downcast_df(df_sampled)
    
    # Optional: Recalculate 'price_ret' to ensure clean data for the app
    if 'price_ret' not in df_optimized.columns and 'price' in df_optimized.columns:
        df_optimized['price_ret'] = df_optimized['price'].pct_change().fillna(0)
    
    # 4. Save to a new compressed file
    print(f"3. Saving optimized sample to {OUTPUT_FILE}...")
    
    df_optimized.to_csv(OUTPUT_FILE, index=False, compression='gzip')

    print("\n-------------------------------------------------")
    print("✅ SUCCESS!")
    print(f"Optimized file created: {OUTPUT_FILE}")
    print(f"Final Row Count: {len(df_optimized):,}")
    print(f"Disk size (Original): {os.path.getsize(INPUT_FILE) / (1024*1024):.2f} MB")
    print(f"Disk size (Optimized): {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    print("-------------------------------------------------")
    print("Next Step: Use 'bitmex_incremental_book_L2_5M_sample.csv.gz' in your Streamlit app.")

if __name__ == "__main__":
    process_and_save_data()