#Continue training the model with improvements from Document 2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

# --- Configuration ---
class Config:
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-?!&"
    VOCAB_SIZE = len(CHARS) + 1
    
    # Training Parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # Additional epochs to train
    LEARNING_RATE = 1e-3
    
    # Image Size
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    HIDDEN_SIZE = 256
    
    # Directory Structure
    BASE_DIR = os.getcwd()
    CSV_DIR = os.path.join(BASE_DIR, "CSV")
    TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_v2/train")
    
    # Annotation Files
    TRAIN_CSV = os.path.join(CSV_DIR, "written_name_train.csv")
    
    # Model Paths
    MINI_MODEL_PATH = os.path.join(BASE_DIR, "htr_crnn_mini.pth")


def create_char_to_int_mapping(chars: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    char_to_int = {char: i + 1 for i, char in enumerate(chars)}
    int_to_char = {i + 1: char for i, char in enumerate(chars)}
    char_to_int['CTC_BLANK'] = 0
    int_to_char[0] = ''
    return char_to_int, int_to_char

CHAR_TO_INT, INT_TO_CHAR = create_char_to_int_mapping(Config.CHARS)


def load_all_annotations() -> Dict[str, Tuple[List[Tuple[str, str]], str]]:
    csv_path = Config.TRAIN_CSV
    img_dir = Config.TRAIN_IMG_DIR
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found at {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    df.columns = ['image_filename', 'transcription_text']
    df.dropna(inplace=True)
    
    df['exists'] = df['image_filename'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    df = df[df['exists']].drop(columns=['exists'])

    annotations = list(df[['image_filename', 'transcription_text']].itertuples(index=False, name=None))
    print(f"✅ Loaded {len(annotations)} valid records for training.")
    
    return {'train': (annotations, img_dir)}


class HTRDataset(Dataset):
    def __init__(self, img_dir: str, annotations: List[Tuple[str, str]], transform=None):
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform
        self.char_to_int = CHAR_TO_INT

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_filename, label_text = self.annotations[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (Config.IMG_WIDTH, Config.IMG_HEIGHT), color='black')
            
        if self.transform:
            image = self.transform(image)
        
        label_encoded = [self.char_to_int.get(char, 0) for char in label_text]
        target = torch.tensor(label_encoded, dtype=torch.long)
        target_len = torch.tensor(len(target), dtype=torch.long)
        
        return image, target, target_len


def collate_fn(batch, model_cnn_output_width=None):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    max_target_len = max(target_lengths)
    padded_targets = torch.zeros((len(targets), max_target_len), dtype=torch.long)
    for i, target in enumerate(targets):
        padded_targets[i, :target.size(0)] = target
    
    target_lengths = torch.stack(target_lengths)
    
    # Use computed CNN output width if provided, else use approximation
    if model_cnn_output_width is None:
        model_cnn_output_width = Config.IMG_WIDTH // 4
    
    input_lengths = torch.full((len(batch),), model_cnn_output_width, dtype=torch.long)
    
    return images, padded_targets, input_lengths, target_lengths


class CRNN(nn.Module):
    def __init__(self, img_height, vocab_size, hidden_size):
        super(CRNN, self).__init__()
        
        # Improved CNN with deeper architecture and BatchNorm
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # Map from CNN output (512 channels) to RNN input
        self.map_to_rnn = nn.Linear(512, hidden_size)
        
        # Improved RNN with dropout and extra layer
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )
        
        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        rnn_input = self.map_to_rnn(cnn_out)
        rnn_input = rnn_input.permute(1, 0, 2)
        
        rnn_out, _ = self.rnn(rnn_input)
        output = self.linear(rnn_out)
        output = nn.functional.log_softmax(output, dim=2)
        
        return output
    
    def get_cnn_output_width(self, batch_size=1, device='cpu'):
        """Compute actual CNN output width dynamically"""
        with torch.no_grad():
            dummy_input = torch.zeros(batch_size, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH, device=device)
            cnn_out = self.cnn(dummy_input)
            return cnn_out.size(-1)


def train_htr_model(data_loader, model, criterion, optimizer, device, scheduler=None):
    """Training with gradient clipping - returns lowest loss from epoch"""
    model.train()
    min_loss = float('inf')
    
    pbar = tqdm(data_loader, desc="Training")
    for images, targets, input_lengths, target_lengths in pbar:
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets, input_lengths.to(device), target_lengths.to(device))
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        
        batch_loss = loss.item()
        min_loss = min(min_loss, batch_loss)
        pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 'min_loss': f'{min_loss:.4f}'})

    return min_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if model exists
    if not os.path.exists(Config.MINI_MODEL_PATH):
        print(f"❌ No existing model found at {Config.MINI_MODEL_PATH}")
        print("Please train the model first using the training script.")
        exit(1)
    
    # Image transformations with augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
        transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), shear=2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load data
    all_data = load_all_annotations()
    
    if 'train' not in all_data:
        print("❌ Training data not found!")
        exit(1)
    
    train_annotations, train_img_dir = all_data['train']
    train_dataset = HTRDataset(img_dir=train_img_dir, annotations=train_annotations, transform=train_transform)
    
    print("\n" + "="*60)
    print("CONTINUE TRAINING FROM SAVED MODEL (IMPROVED)")
    print("="*60)
    
    mini_size = 5000
    mini_indices = torch.randperm(len(train_dataset))[:mini_size].tolist()
    mini_dataset = Subset(train_dataset, mini_indices)
    
    mini_loader = DataLoader(
        mini_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"Mini dataset size: {len(mini_dataset)} samples")
    
    # Load existing model
    print(f"\nLoading existing model from: {Config.MINI_MODEL_PATH}")
    model_mini = CRNN(Config.IMG_HEIGHT, Config.VOCAB_SIZE, Config.HIDDEN_SIZE).to(device)
    model_mini.load_state_dict(torch.load(Config.MINI_MODEL_PATH, map_location=device))
    print(f"✅ Model loaded successfully!")
    
    # Compute actual CNN output width for accurate CTC loss
    cnn_output_width = model_mini.get_cnn_output_width(batch_size=1, device=device)
    print(f"CNN output width: {cnn_output_width}")
    
    # Update collate function to use actual CNN output width
    def collate_fn_with_width(batch):
        return collate_fn(batch, model_cnn_output_width=cnn_output_width)
    
    mini_loader = DataLoader(
        mini_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_with_width
    )
    
    print(f"\nContinuing training for {Config.NUM_EPOCHS} more epochs...\n")
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer_mini = optim.Adam(model_mini.parameters(), lr=Config.LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_mini, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    best_loss_mini = float('inf')
    loss_history = []
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
        min_loss = train_htr_model(mini_loader, model_mini, criterion, optimizer_mini, device, scheduler)
        print(f"Minimum Loss: {min_loss:.4f}\n")
        
        loss_history.append(min_loss)
        
        # Step learning rate based on loss plateau
        scheduler.step(min_loss)
        
        if min_loss < best_loss_mini:
            best_loss_mini = min_loss
            torch.save(model_mini.state_dict(), Config.MINI_MODEL_PATH)
            print(f"✅ Best model saved! Lowest loss: {best_loss_mini:.4f}\n")
        else:
            print(f"Loss did not improve. Best so far: {best_loss_mini:.4f}\n")
    
    print(f"\n✅ Continued training complete! Best loss achieved: {best_loss_mini:.4f}")
    print(f"Model saved to: {Config.MINI_MODEL_PATH}")
    print(f"\nLoss history by epoch:")
    for epoch, loss in enumerate(loss_history, 1):
        print(f"  Epoch {epoch}: {loss:.4f}")
