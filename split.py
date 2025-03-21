# split_csv.py - Run this locally to split your large CSV file

import pandas as pd
import os
import sys

def split_csv(input_file, output_dir="data", chunk_size=100000):
    """
    Split a large CSV file into smaller chunks.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the split files
        chunk_size: Number of rows in each chunk
    """
    print(f"Splitting file: {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the file in chunks to avoid loading it all into memory
    chunk_counter = 1
    total_rows = 0
    
    # Read and process chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Create output filename with padded counter
        output_file = os.path.join(output_dir, f"part_{chunk_counter:02d}.csv")
        
        # Save this chunk with headers
        chunk.to_csv(output_file, index=False)
        
        # Update counters
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        # Print progress
        chunk_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Created {output_file}: {rows_in_chunk} rows, {chunk_size_mb:.2f} MB")
        
        chunk_counter += 1
    
    print(f"\nSplitting complete!")
    print(f"Total rows processed: {total_rows}")
    print(f"Total files created: {chunk_counter - 1}")
    print(f"Files are saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # Get input file from command line argument or use default
    input_file = "data/final_adjusted_stock_data.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # For 7 million rows in 715MB file, we want ~10 files of ~70MB each
    # 7,000,000 ÷ 10 = 700,000 rows per file
    chunk_size = 700000
    if len(sys.argv) > 2:
        chunk_size = int(sys.argv[2])
    
    # Run the splitter
    split_csv(input_file, chunk_size=chunk_size)