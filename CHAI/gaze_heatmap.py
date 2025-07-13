import json
import matplotlib.pyplot as plt
import numpy as np
import os

def get_gaze_freq(gaze_log):
    freq = {}
    for frame in gaze_log:
        if frame["sector"] is not None:
            sector = frame["sector"]
            if sector not in freq:
                freq[sector] = 0
            freq[sector] += 1
    return freq

# Only try to load gaze log if it exists
if os.path.exists("./frames/gaze_log.json"):
    try:
        with open("./frames/gaze_log.json", "r") as f:
            gaze_log = json.load(f)
        print("Gaze log loaded successfully")
        freq = get_gaze_freq(gaze_log)
        print("Frequency distribution:", freq)
    except Exception as e:
        print(f"Error loading gaze log: {e}")
        gaze_log = []
        freq = {}
else:
    print("Gaze log file not found, skipping initialization")
    gaze_log = []
    freq = {}

# Only create heatmap if we have data
if freq and gaze_log:
    # Create a 3x3 grid for the heatmap
    # Sectors are typically numbered 0-8 in a 3x3 grid
    heatmap_data = np.zeros((3, 3))

    # Map sectors to grid positions
    # Assuming sectors are numbered 0-8 in a 3x3 grid
    for sector, count in freq.items():
        if 1 <= sector <= 9:
            idx = sector - 1
            row = 2 - (idx // 3)  # Mirror vertically
            col = idx % 3
            heatmap_data[row, col] = count

    print("Heatmap data:")
    print(heatmap_data)

    # Create the heatmap visualization
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(heatmap_data, cmap='hot_r', interpolation='nearest')
    plt.colorbar(heatmap, label='Gaze Frequency')

    # Add text annotations to show exact values
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{int(heatmap_data[i, j])}', 
                    ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    plt.title('Gaze Heatmap (3x3 Grid)', fontsize=14, fontweight='bold')
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)

    # Set tick labels
    plt.xticks([0, 1, 2], ['Left', 'Center', 'Right'])
    plt.yticks([0, 1, 2], ['Top', 'Middle', 'Bottom'])

    plt.tight_layout()
    
    # Create frames directory if it doesn't exist
    os.makedirs("./frames", exist_ok=True)
    
    plt.savefig('./frames/gaze_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Heatmap saved as './frames/gaze_heatmap.png'")
else:
    print("No gaze data available for heatmap generation")