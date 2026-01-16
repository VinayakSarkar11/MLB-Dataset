import os       
import cv2     
import torch
import numpy as np
import random

def inspect_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    all_videos = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".mp4"):
                all_videos.append(os.path.join(root, file))

    video_path = random.choice(all_videos)
    print(f"🎥 Inspecting: {os.path.basename(video_path)}")

    # Read Video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        #ret gives if frame exists; frame gives data
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert Open CV's BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Convert video from 1080p to 112x112 pixels 
        frame = cv2.resize(frame, (112, 112))
        
        frames.append(frame)
    
    cap.release()
    
    #CONVERT TO TENSOR - Current Shape: (Time, Height, Width, Channels)
    buffer = np.array(frames)
    video_tensor = torch.from_numpy(buffer)

    # Convert tensor to pytorch format - (Time, Channels, Height, Width) 
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    
    # Normalize RGB on 0-1 scale
    video_tensor = video_tensor.float() / 255.0

    print("\n SUCCESS! The AI can see this video.")
    print(f"📊 Tensor Shape: {video_tensor.shape}")
    print(f"   - [0] Time (Frames):   {video_tensor.shape[0]}")
    print(f"   - [1] Channels (RGB):  {video_tensor.shape[1]}")
    print(f"   - [2] Height:          {video_tensor.shape[2]}")
    print(f"   - [3] Width:           {video_tensor.shape[3]}")

    #Ensure vides are of normalized length.
    duration_sec = video_tensor.shape[0] / 60 #assume 60 fps
    print(f"⏱️ Estimated Duration: {duration_sec:.2f} seconds")

    if video_tensor.shape[0] < 10:
        print("⚠️ WARNING: This video is too short. It might just be a glitch.")
    elif video_tensor.shape[0] > 150:
        print("⚠️ WARNING: This video is long. We will need to trim it later.")
    else:
        print("Perfect length for training.")

if __name__ == "__main__":
    inspect_dataset()