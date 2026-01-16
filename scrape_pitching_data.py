import json
import os
import yt_dlp
import sys
import re
import shutil
import subprocess
from collections import defaultdict

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, "mlb-youtube-segmented.json")
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_full")
BALANCED_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset_balanced")
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp_games")
LOG_FILE = os.path.join(SCRIPT_DIR, "download_errors.log")

# Ensure folders exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BALANCED_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def log_error(video_id, message):
    """Writes errors to a file so we don't miss them in the terminal scroll"""
    print(f"   [Log] {video_id}: {message}")
    with open(LOG_FILE, "a") as f:
        f.write(f"{video_id}: {message}\n")

def load_dataset_items(filename="mlb-youtube-segmented.json"):
    """Finds and loads the JSON file, returning a standard list of items."""
    json_full_path = os.path.join(SCRIPT_DIR, filename)

    if not os.path.exists(json_full_path):
        print(f"Error: {json_full_path} not found.")
        sys.exit(1)

    print(f"Reading JSON from {filename}...")
    with open(json_full_path, 'r') as f:
        data = json.load(f)
    
    # Normalize List vs Dict structure
    if isinstance(data, dict):
        return list(data.values())
    return data

def get_video_id(url):
    """Extracts YouTube ID from various URL formats."""
    if "=" in url: return url.split('=')[-1]
    elif "youtu.be" in url: return url.split('/')[-1]
    else: return url[-11:]

def clean_folder_name(name):
    """Turns 'Four-Seam Fastball' into 'four_seam_fastball'."""
    if not name: return "unknown"
    return re.sub(r'[^a-zA-Z0-9]', '_', name).lower()

def get_video_duration(file_path):
    """Returns duration in seconds using ffprobe (Critical for verifying Full Game downloads)."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except:
        return 0.0

def check_and_fix_file(final_path):
    """Check if file is a ghost file"""
    
    if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
        return True

    ghost_path = final_path + ".part"
    if os.path.exists(ghost_path):
        size = os.path.getsize(ghost_path)
        if size > 1000:
            try:
                os.rename(ghost_path, final_path)
                print(f"   Ghost file found & fixed: {os.path.basename(final_path)}")
                return True
            except OSError as e:
                print(f"   Could not rename ghost file: {e}")
                return False
    
    return False

def perform_individual_download(entry, output_folder, file_prefix, ydl_opts):
    video_id = get_video_id(entry['url'])
    final_filename = os.path.join(output_folder, f"{file_prefix}_{video_id}.mp4")

    if check_and_fix_file(final_filename):
        return True

    try:
        current_opts = ydl_opts.copy()
        current_opts['outtmpl'] = final_filename
        
        start = entry.get('start')
        end = entry.get('end')
        
        if start is None or end is None:
            log_error(video_id, "No timestamps in JSON")
            return False

        current_opts['download_ranges'] = lambda _, __: [{'start_time': start, 'end_time': end}]
        
        with yt_dlp.YoutubeDL(current_opts) as ydl:
            ydl.download([entry['url']])
            
        if check_and_fix_file(final_filename):
            return True
        else:
            log_error(video_id, "Download finished but file is missing/empty")
            return False

    except Exception as e:
        log_error(video_id, str(e))
        return False

def download_balanced_dataset(limit_per_class=10):
    items = load_dataset_items()
    print(f"Starting BALANCED download (Limit: {limit_per_class})...")
    print(f"Output: {BALANCED_OUTPUT_DIR}")

    TARGETS = {'fastball': limit_per_class, 'slider': limit_per_class, 'curveball': limit_per_class, 'sinker': limit_per_class}
    counts = {k: 0 for k in TARGETS}
    SHORT_NAMES = {'fastball': 'FF', 'slider': 'SL', 'curveball': 'CU', 'sinker': 'SI'}

    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'no_warnings': True,
        'cookiesfrombrowser': ('chrome',), 
    }

    for entry in items:
        if all(counts[k] >= TARGETS[k] for k in TARGETS):
            print("\nBalanced Dataset Complete.")
            break

        pitch_type = entry.get('type')
        if pitch_type in TARGETS and counts[pitch_type] < TARGETS[pitch_type]:
            short_name = SHORT_NAMES[pitch_type]
            folder = os.path.join(BALANCED_OUTPUT_DIR, short_name)
            os.makedirs(folder, exist_ok=True)
            
            print(f"[{counts[pitch_type]+1}/{TARGETS[pitch_type]}] Downloading {short_name}...", end="\r")
            if perform_individual_download(entry, folder, short_name, ydl_opts):
                counts[pitch_type] += 1

# Download full dataset
def download_full_dataset():
    items = load_dataset_items()
    print(f"Starting FULL BATCH download ({len(items)} clips)...")
    print(f"Output: {BASE_OUTPUT_DIR}")
    
    #Group pitches by game
    games = defaultdict(list)
    for entry in items:
        vid_id = get_video_id(entry['url'])
        games[vid_id].append(entry)

    print(f"Organized into {len(games)} unique source videos.")

    #Process game
    for vid_idx, (video_id, clips) in enumerate(games.items()):
        missing_clips = []
        for entry in clips:
            pitch_type = entry.get('type')
            if not pitch_type: continue
            folder = clean_folder_name(pitch_type)
            segment_id = f"{video_id}_{int(entry['start'])}"
            final_path = os.path.join(BASE_OUTPUT_DIR, folder, f"{folder}_{segment_id}.mp4")
            if not check_and_fix_file(final_path):
                missing_clips.append(entry)
        
        if not missing_clips:
            if vid_idx % 5 == 0: print(f"[{vid_idx}/{len(games)}] Skipping {video_id} (All clips exist)...", end="\r")
            continue

        print(f"\n[{vid_idx}/{len(games)}] Processing Game {video_id} ({len(missing_clips)} needed clips)...")
        
        full_game_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")
        if not os.path.exists(full_game_path):
            print(f"   Downloading source video (may take time)...")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': full_game_path,
                'quiet': True,
                'no_warnings': True,
                'cookiesfrombrowser': ('chrome',),
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            except Exception as e:
                print(f"   Source download failed: {e}")
                continue

        total_duration = get_video_duration(full_game_path)
        success_count = 0
        
        for entry in missing_clips:
            pitch_type = entry.get('type')
            folder = clean_folder_name(pitch_type)
            folder_path = os.path.join(BASE_OUTPUT_DIR, folder)
            os.makedirs(folder_path, exist_ok=True)
            
            start = entry['start']
            end = entry['end']
            
            if start > total_duration: continue # Timestamp out of bounds

            segment_id = f"{video_id}_{int(start)}"
            final_filename = os.path.join(folder_path, f"{folder}_{segment_id}.mp4")
            
            duration = end - start
            
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", full_game_path,
                "-t", str(duration), "-c:v", "libx264", "-c:a", "aac",
                "-loglevel", "error", final_filename
            ]
            try:
                subprocess.run(cmd, check=True)
                if check_and_fix_file(final_filename):
                    success_count += 1
            except subprocess.CalledProcessError:
                pass

        print(f"   Extracted {success_count} clips.")
        
        if os.path.exists(full_game_path):
            os.remove(full_game_path)

def audit_dataset():
    if not os.path.exists(BASE_OUTPUT_DIR):
        print("'dataset_full' folder not found.")
        return

    print(f"\nAUDIT RESULTS ({BASE_OUTPUT_DIR}):\n")
    print(f"{'PITCH TYPE':<20} | {'FILES':<10} | {'SIZE (MB)':<10}")
    print("-" * 45)

    total_files = 0
    total_size = 0

    for pitch_type in sorted(os.listdir(BASE_OUTPUT_DIR)):
        folder_path = os.path.join(BASE_OUTPUT_DIR, pitch_type)
        if not os.path.isdir(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
        valid_files = []
        folder_size = 0
        
        for f in files:
            path = os.path.join(folder_path, f)
            size = os.path.getsize(path)
            if size > 1000:
                valid_files.append(f)
                folder_size += size
        
        size_mb = folder_size / (1024 * 1024)
        print(f"{pitch_type:<20} | {len(valid_files):<10} | {size_mb:.2f} MB")
        total_files += len(valid_files)
        total_size += size_mb

    print("-" * 45)
    print(f"TOTAL VALID CLIPS: {total_files}")
    print(f"TOTAL SIZE:        {total_size:.2f} MB")

if __name__ == "__main__":
    # download_balanced_dataset(limit_per_class=10) 
    download_full_dataset()
    audit_dataset()