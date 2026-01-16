import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

#determine start of pitch by two factors - green (grass) present in shot, high motion
def is_scene_change(frame1, frame2, threshold=0.6):
    """
    Compares color histograms to detect hard camera cuts.
    """
    try:
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation < threshold
    except Exception:
        return False

def calculate_scene_motion(frames):
    """
    Calculates the average motion energy of a sequence of frames.
    Used to distinguish 'Static Runner' (Low) from 'Pitching' (High).
    """
    if len(frames) < 2: return 0.0
    
    score = 0.0
    h, w, _ = frames[0].shape
    sh, eh = int(h*0.25), int(h*0.75)
    sw, ew = int(w*0.25), int(w*0.75)

    for i in range(1, len(frames)):
        prev = cv2.GaussianBlur(cv2.cvtColor(frames[i-1][sh:eh, sw:ew], cv2.COLOR_RGB2GRAY), (21, 21), 0)
        curr = cv2.GaussianBlur(cv2.cvtColor(frames[i][sh:eh, sw:ew], cv2.COLOR_RGB2GRAY), (21, 21), 0)
        
        # Diff
        diff = cv2.absdiff(prev, curr)
        score += np.mean(diff)
        
    return score / len(frames)

def find_the_field(frames, green_threshold=0.15):
    """
    Scans the video for Green Scenes and picks the one with HIGHEST MOTION.
    """
    if not frames: return 0
    
    # 1. Map out "Green" frames
    lower_green = np.array([35, 40, 40]) 
    upper_green = np.array([85, 255, 255])
    is_field = []
    
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        ratio = np.count_nonzero(mask) / (frame.shape[0] * frame.shape[1])
        is_field.append(ratio > green_threshold)

    scenes = [] # (start_index, end_index)
    current_start = -1
    
    for i in range(len(frames)):
        cut = (i > 0 and is_scene_change(frames[i-1], frames[i]))
        
        if is_field[i] and not cut:
            if current_start == -1: current_start = i
        else:
            if current_start != -1:
                scenes.append((current_start, i))
                current_start = -1
    
    if current_start != -1: scenes.append((current_start, len(frames)))

    if not scenes: return 0
    
    best_start = 0
    max_energy = -1.0
    
    for start, end in scenes:
        if (end - start) < 10: continue
            
        scene_frames = frames[start:end]
        energy = calculate_scene_motion(scene_frames)
        
        if energy > max_energy:
            max_energy = energy
            best_start = start
            
    return best_start

class MLBDataset(Dataset):
    def __init__(self, dataset_dir, frames_per_clip=16):
        self.frames_per_clip = frames_per_clip
        self.samples = []
        
        if not os.path.exists(dataset_dir):
            print(f"Error: Dataset directory '{dataset_dir}' not found.")
            self.classes = []
        else:
            self.classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        
        print(f"🔍 Found {len(self.classes)} classes: {self.classes}")

        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(dataset_dir, class_name)
            files = [f for f in os.listdir(class_path) if f.endswith(".mp4")]
            for file in files:
                self.samples.append((os.path.join(class_path, file), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            try:
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            except Exception:
                continue
        cap.release()
        
        if len(frames) == 0:
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx) 

        if len(frames) < self.frames_per_clip:
            frames = frames * (self.frames_per_clip // len(frames) + 1)

        start_frame = find_the_field(frames)
        
        max_start = max(0, len(frames) - self.frames_per_clip)
        if start_frame > max_start: start_frame = max_start
            
        active_frames = frames[start_frame:]
        
        indices = np.linspace(0, len(active_frames) - 1, self.frames_per_clip).astype(int)
        sampled_frames = [active_frames[i] for i in indices]

        if random.random() > 0.5:
            sampled_frames = [cv2.flip(f, 1) for f in sampled_frames]

        tensor = torch.from_numpy(np.array(sampled_frames))
        tensor = tensor.permute(3, 0, 1, 2).float() / 255.0
        
        return tensor, label