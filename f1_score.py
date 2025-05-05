import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# get all csv from ./hasil
import os
import glob
import sys
import warnings
warnings.filterwarnings("ignore")

# Get the list of all CSV files in the directory
csv_files = glob.glob('./hasil/*.csv')

if len(csv_files) == 0:
    print("No CSV files found in the 'hasil' directory.")
    # Optionally exit or handle this case differently
    # sys.exit(1) # Removing exit so the rest of script doesn't fail if run non-interactively

# Process each CSV file
for file in csv_files:
    print(f"Processing file: {file}")

    try:
        # Load the data from the CSV file
        df = pd.read_csv(file)

        # Determine RFB status from filename
        filename_base = os.path.basename(file)
        rfb_status = ""
        # Check if 'rfb' is in the filename (case-insensitive)
        if 'rfb' in filename_base.lower():
             rfb_status = "RFB"
        # You might want more robust logic here if the naming convention is complex

        # Get scale from DataFrame or default to 1.0
        # Assuming 'scale' column exists and is consistent within a file
        scale = df['scale'].iloc[0] if 'scale' in df.columns and not df['scale'].empty else 1.0

        # Ensure 'tilt', 'pan', and 'f1_score' columns exist
        if not all(col in df.columns for col in ['tilt', 'pan', 'f1_score']):
            print(f"Skipping {file}: Missing one or more required columns (tilt, pan, f1_score).")
            continue

        # Create a pivot table for the heatmap
        # Using aggfunc='mean' in case of duplicate pan/tilt pairs, though not strictly necessary if data is unique
        pivot_table = df.pivot_table(values='f1_score', index='tilt', columns='pan', aggfunc='mean')

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        # Change cmap="viridis" to cmap="Greys" for grayscale
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="Greys", cbar_kws={'label': 'F1 Score'})

        # Construct title
        title = f"F1 Score Heatmap (Grayscale) - Scaled to {scale}px" # Added Grayscale to title
        if rfb_status:
             title += f" - {rfb_status}"
        plt.title(title)

        plt.xlabel('Pan Angle')
        plt.ylabel('Tilt Angle')

        # Invert y-axis to match the order in the data or typical orientation
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save the heatmap as a PNG file
        # Construct output filename based on original filename, scale, and rfb status
        # Replace .csv with .png and insert scale/rfb info before extension
        output_filename_base, ext = os.path.splitext(filename_base)
        output_filename = f"hasil/{output_filename_base}_scale{scale}_grayscale" # Added _grayscale to filename
        if rfb_status:
             output_filename += f"_{rfb_status}"
        output_filename += ".png"


        plt.savefig(output_filename)
        plt.close() # Close the plot to free memory

        print(f"Grayscale F1 score heatmap saved as '{output_filename}'")

    except Exception as e:
        print(f"Error processing file {file}: {e}")

print("Finished processing all CSV files in ./hasil.")