import pandas as pd
import os

input_file = 'bitmex_incremental_book_L2.csv'
output_file = 'bitmex_incremental_book_L2.csv.gz'

print(f"Reading {input_file} as GZIP...")

# --- THE KEY CHANGE: FORCE PANDAS TO DECOMPRESS THE INPUT FILE ---
# 1. compression='gzip': Forces pandas to use zlib to decompress the input.
# 2. encoding='utf-8': Use standard encoding after decompression (you can try 'latin1' if utf-8 fails later).
# 3. sep=',': We are assuming it's a standard CSV once decompressed.
try:
    df_original = pd.read_csv(input_file, 
                              compression='gzip', 
                              encoding='utf-8', 
                              sep=',') 

    print(f"Original DataFrame loaded successfully. Shape: {df_original.shape}")
    print(f"Compressing and saving to {output_file}...")

    # Save to the new file, ensuring it is also compressed (if needed)
    df_original.to_csv(output_file, index=False, compression='gzip')

    print(f"Compression complete.")
    print(f"Original file size (disk): {os.path.getsize(input_file) / (1024*1024):.2f} MB")
    print(f"Compressed output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to load file as GZIP-CSV.")
    print(f"Details: {e}")
    print("If this fails, the file is either corrupt or encrypted/compressed with a different method.")