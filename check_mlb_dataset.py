import os
import cv2
import numpy as np
from mlb_dataset import MLBDataset #import MLBDataset class

"""Check MLBDataset correctly identifies pitching frames and does not excessively cut video"""

def verify_logic():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    
    print("Initializing Dataset...")
    dataset = MLBDataset(dataset_dir, frames_per_clip=16)

    video_path, label_idx = dataset.samples[1]
    video_filename = os.path.basename(video_path)
    print(f"Analyzing Video: {video_filename}")

    print("Loading sample #1...")
    tensor, label = dataset[1] 
    
    print(f"Tensor Shape: {tensor.shape}")
    
    # Convert Tensor back to Images
    images = tensor.permute(1, 2, 3, 0).numpy() 
    images = (images * 255).astype(np.uint8)    # Denormalize 
    
    # Create a grid (2 rows of 8 frames)
    top_row = np.hstack(images[:8])
    bot_row = np.hstack(images[8:])
    grid = np.vstack((top_row, bot_row))
    
    # RGB -> BGR for OpenCV saving
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    
    output_path = os.path.join(script_dir, "visual_check.jpg")
    cv2.imwrite(output_path, grid)
    
    print(f"Saved inspection image to: {output_path}")

if __name__ == "__main__":
    verify_logic()