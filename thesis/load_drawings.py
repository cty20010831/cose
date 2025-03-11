"""
This python script is used to load each participant's drawing (for all the three 
groups of drawings) in ndjson and store them into three separate large ndjson 
files containing all participants' drawings in the respective group.
"""

import os
import json
import glob

DRAWING_DIR = "../../../data/drawings/ndjson"
OUTPUT_DIR = "data"

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Output files for each group
    group_files = {
        'A': os.path.join(OUTPUT_DIR, 'raw_group_A_drawings.ndjson'),
        'B': os.path.join(OUTPUT_DIR, 'raw_group_B_drawings.ndjson'),
        'C': os.path.join(OUTPUT_DIR, 'raw_group_C_drawings.ndjson')
    }
    
    # Initialize output files (clear existing content)
    for group_file in group_files.values():
        with open(group_file, 'w') as f:
            pass
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(DRAWING_DIR, "batch_*"))
    print(f"Found {len(batch_dirs)} batch directories")
    
    for batch_dir in batch_dirs:
        print(f"Processing {os.path.basename(batch_dir)}")
        
        # Process all ndjson files in the batch directory
        drawing_files = glob.glob(os.path.join(batch_dir, "*.ndjson"))
        
        for drawing_file in drawing_files:
            filename = os.path.basename(drawing_file)
            
            # Extract group identifier from filename (e.g., A1HGOFQI8TQCWB_Group_A.ndjson -> A)
            if "_Group_" in filename:
                group = filename.split("_Group_")[1].split(".")[0]
                
                # Check if it's one of our target groups
                if group in group_files:
                    # Read the drawing data
                    with open(drawing_file, 'r') as src_file:
                        drawing_data = src_file.read()
                    
                    # If file has content, append to the appropriate group file
                    if drawing_data.strip():
                        with open(group_files[group], 'a') as dest_file:
                            dest_file.write(drawing_data)
                            # If the drawing data doesn't end with a newline, add one
                            if not drawing_data.endswith('\n'):
                                dest_file.write('\n')
                            
                        print(f"Added {filename} to Group {group}")
                else:
                    print(f"Skipping {filename} - unknown group: {group}")
            else:
                print(f"Skipping {filename} - doesn't match expected format")
    
    # Print summary
    for group, file_path in group_files.items():
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"Group {group}: {line_count} drawings saved to {file_path}")
        else:
            print(f"Group {group}: No drawings found")

if __name__ == "__main__":
    main()

# Run the script (under thesis directory)
# python load_drawings.py