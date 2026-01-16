import os
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from sklearn.metrics import classification_report, confusion_matrix
from mlb_dataset import MLBDataset 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time

BATCH_SIZE = 2   
#accumulation steps allow us to effectively increase batch size without overwhelming CPU
ACCUMULATION_STEPS = 8 
EPOCHS = 8           
LEARNING_RATE = 1e-4  
FRAMES_PER_CLIP = 32
MODEL_FILENAME = "baseball_model.pth"
SEED = 42 


def set_seed(seed=42):
    """Forces the random split to be identical every run."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def split_by_game_id(full_dataset, train_ratio=0.8, seed=42):
    #Splits data by game 
    game_to_indices = {}
    
    for idx, (path, label) in enumerate(full_dataset.samples):
        filename = os.path.basename(path)
        try:
            parts = filename.split('_')
            video_id = parts[-2] 
        except IndexError:
            video_id = "unknown"
        
        if video_id not in game_to_indices:
            game_to_indices[video_id] = []
        game_to_indices[video_id].append(idx)
        
    unique_games = list(game_to_indices.keys())
    random.Random(seed).shuffle(unique_games)
    
    n_games = len(unique_games)
    n_train = int(n_games * train_ratio)
    
    train_games = unique_games[:n_train]
    val_games = unique_games[n_train:]
    
    train_idx = [i for g in train_games for i in game_to_indices[g]]
    val_idx = [i for g in val_games for i in game_to_indices[g]]
    
    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    
    print(f"   - Unique Games: {n_games}")
    print(f"   - Train: {len(train_games)} games ({len(train_data)} clips)")
    print(f"   - Val:   {len(val_games)} games ({len(val_data)} clips)")
    
    return train_data, val_data

def get_model(num_classes):

    # Load the pre-trained model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    
    #Freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only temporal features
    for param in model.blocks[5].parameters():
        param.requires_grad = True
        
    # Drop certain features to ensure model can't rely on the same feature every time
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(p=0.5), 
        nn.Linear(in_features=2048, out_features=num_classes)
    )
    # Ensure the head is trainable
    for param in model.blocks[5].parameters():
        param.requires_grad = True

    model.blocks[5].pool = nn.AdaptiveAvgPool3d(1)
    
    return model

def train_engine():
    torch.set_num_threads(1) 

    set_seed(SEED)

    #Use optimal hardware
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS Acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset_full")
    save_path = os.path.join(script_dir, MODEL_FILENAME)
    
    full_dataset = MLBDataset(dataset_dir, frames_per_clip=FRAMES_PER_CLIP)
    
    if len(full_dataset) == 0:
        print("Dataset is empty. Run the downloader first.")
        return
    
    print(f" Found {len(full_dataset)} clips across {len(full_dataset.classes)} classes:")
    print(f"   Classes: {full_dataset.classes}")

    # Train/testing done by game
    train_data, val_data = split_by_game_id(full_dataset, train_ratio=0.8, seed=SEED)
    
    #Weighted sampling to avoid fastball imbalance
    all_labels = [label for _, label in full_dataset.samples]
    
    train_indices = train_data.indices
    train_labels = [all_labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_labels)
    class_counts = np.maximum(class_counts, 1)
    
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[t] for t in train_labels]
    
    # Create Sampler
    generator = torch.Generator()
    generator.manual_seed(SEED)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), generator=generator)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    #setup model
    model = get_model(num_classes=len(full_dataset.classes))
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    best_acc = 0.0
    if os.path.exists(save_path):
        print(f"Found existing model: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                best_acc = checkpoint.get('accuracy', 0.0)
                
                # Load optimizer state to preserve momentum 
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded.")
                else:
                    print("⚠️ No optimizer state found in checkpoint.")
                
                print(f"Resuming from Best Val Accuracy: {best_acc:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print("Legacy weights loaded. Starting baseline.")
        except Exception as e:
            print(f"Starting from scratch (Load failed: {e})")

    #train model
    print(f"Training with effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    for epoch in range(EPOCHS):
        model.train() 
        running_loss = 0.0
        
        # CHANGED: Zero grad before loop starts
        optimizer.zero_grad()
        
        for i, (videos, labels) in enumerate(train_loader):

            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            #Only step optimizer every N batches
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUMULATION_STEPS
            
            if i % 10 == 0:
                print(f"   Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item()*ACCUMULATION_STEPS:.4f}", end="\r")

        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Complete | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            #Save optimizer state along with model weights
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc
            }, save_path)
            print(f"   💾 New Best Model Saved! ({best_acc:.2f}%)")

    print(f"\n🏆 Final Best Accuracy: {best_acc:.2f}%")

def evaluate():
    set_seed(SEED)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset_full")
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    print("Loading Data...")
    full_dataset = MLBDataset(dataset_dir, frames_per_clip=FRAMES_PER_CLIP)
    
    print("Recreating Game ID Split for Evaluation...")
    _, val_data = split_by_game_id(full_dataset, train_ratio=0.8, seed=SEED)
    
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Evaluating on {len(val_data)} validation clips.")

    # 3. Load Model
    model = get_model(len(full_dataset.classes))
    model = model.to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint (Acc: {checkpoint.get('accuracy', 0):.2f}%)")
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Model file not found!")
        return

    print("Running Inference...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (videos, labels) in enumerate(val_loader):
            print(f"   Batch {i}/{len(val_loader)}...", end="\r")
            videos = videos.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    
    # Text Report
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_dataset.classes, 
                yticklabels=full_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(script_dir, "confusion_matrix.png"))
    print(f"📸 Saved Confusion Matrix to {os.path.join(script_dir, 'confusion_matrix.png')}")

if __name__ == "__main__":
   train_engine()
    #evaluate()
