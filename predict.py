import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import csv
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

# --- Configuration ---
class Config:
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-?!&"
    VOCAB_SIZE = len(CHARS) + 1
    
    # Image Size
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    HIDDEN_SIZE = 256
    
    # Directory Structure
    BASE_DIR = os.getcwd()
    CSV_DIR = os.path.join(BASE_DIR, "CSV")
    TEST_IMG_DIR = os.path.join(BASE_DIR, "test_v2/test")
    
    # Annotation Files
    TEST_CSV = os.path.join(CSV_DIR, "written_name_test.csv")
    
    # Model Paths
    MINI_MODEL_PATH = os.path.join(BASE_DIR, "htr_crnn_mini.pth")


def create_char_to_int_mapping(chars: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    char_to_int = {char: i + 1 for i, char in enumerate(chars)}
    int_to_char = {i + 1: char for i, char in enumerate(chars)}
    char_to_int['CTC_BLANK'] = 0
    int_to_char[0] = ''
    return char_to_int, int_to_char

CHAR_TO_INT, INT_TO_CHAR = create_char_to_int_mapping(Config.CHARS)


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
        
        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


def decode_ctc(output: torch.Tensor, int_to_char: Dict[int, str]) -> tuple:
    """
    Decodes CTC output using greedy decoding.
    
    Args:
        output: Tensor of shape [SequenceLength, VocabSize] with log probabilities
        int_to_char: Mapping from integer indices to characters
        
    Returns:
        Tuple of (decoded_text, confidence_score)
    """
    probs = output.exp()
    preds = output.argmax(dim=1)
    max_probs = probs.max(dim=1)[0]
    
    decoded_text = []
    confidence_scores = []
    prev_idx = -1
    
    for i, char_idx in enumerate(preds.cpu().numpy()):
        if char_idx != 0 and char_idx != prev_idx:
            decoded_text.append(int_to_char.get(int(char_idx), '?'))
            confidence_scores.append(max_probs[i].item())
        prev_idx = char_idx
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    return "".join(decoded_text), avg_confidence


def predict_handwritten_text(image_path: str, model: nn.Module, device) -> tuple:
    """
    Predicts text from a handwritten image using a loaded model.
    
    Args:
        image_path: Path to the image file to predict on
        model: Already loaded CRNN model
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_text, confidence_score)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    try:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            output = output.squeeze(1)
            predicted_text, confidence = decode_ctc(output, INT_TO_CHAR)
        
        return predicted_text, confidence
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_batch(model_path: str, image_dir: str, output_dir: str = "output", save_csv: bool = True):
    """
    Predicts text from all images in a directory using the mini model.
    
    Args:
        model_path: Path to the saved model weights
        image_dir: Directory containing images to predict on
        output_dir: Directory to save output results
        save_csv: Whether to save results as CSV file
    """
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading CRNN model...")
    model = CRNN(Config.IMG_HEIGHT, Config.VOCAB_SIZE, Config.HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = sorted([f for f in os.listdir(image_dir) 
                          if os.path.splitext(f)[1].lower() in image_extensions])
    
    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images. Starting predictions...\n")
    
    results = []
    
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, filename)
        
        try:
            predicted_text, confidence = predict_handwritten_text(image_path, model, device)
            status = "✅"
            print(f"{status} [{idx}/{len(image_files)}] {filename}")
            print(f"   Predicted: '{predicted_text}' (Confidence: {confidence:.4f})\n")
            
            results.append({
                'filename': filename,
                'predicted_text': predicted_text,
                'confidence': f"{confidence:.4f}"
            })
            
        except (FileNotFoundError, RuntimeError) as e:
            print(f"❌ [{idx}/{len(image_files)}] {filename}")
            print(f"   Error: {e}\n")
            
            results.append({
                'filename': filename,
                'predicted_text': 'ERROR',
                'confidence': 'N/A'
            })
    
    # Save results to CSV
    if save_csv:
        csv_path = os.path.join(output_dir, "predictions.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'predicted_text', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ Results saved to: {csv_path}")
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
    
    # Save results to text file
    txt_path = os.path.join(output_dir, "predictions.txt")
    try:
        with open(txt_path, 'w', encoding='utf-8') as txtfile:
            txtfile.write("HTR CRNN Model - Predictions\n")
            txtfile.write("=" * 60 + "\n\n")
            for result in results:
                txtfile.write(f"File: {result['filename']}\n")
                txtfile.write(f"Predicted Text: {result['predicted_text']}\n")
                txtfile.write(f"Confidence: {result['confidence']}\n")
                txtfile.write("-" * 60 + "\n")
        print(f"✅ Results saved to: {txt_path}")
    except Exception as e:
        print(f"❌ Failed to save TXT: {e}")
    
    print(f"\n✅ Prediction complete! Results saved in '{output_dir}' folder")


def predict_single_image(model_path: str, image_path: str):
    """
    Predict text from a single image.
    
    Args:
        model_path: Path to the saved model weights
        image_path: Path to the image file
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading CRNN model...")
    model = CRNN(Config.IMG_HEIGHT, Config.VOCAB_SIZE, Config.HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully\n")
    
    try:
        predicted_text, confidence = predict_handwritten_text(image_path, model, device)
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Text: '{predicted_text}'")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("CRNN MODEL - TESTING/INFERENCE")
    print("=" * 60 + "\n")
    #'''
    # Option 1: Batch prediction on test directory
    img_dir = Config.TEST_IMG_DIR
    output_folder = "output"
    
    predict_batch(
        model_path=Config.MINI_MODEL_PATH,
        image_dir=img_dir,
        output_dir=output_folder,
        save_csv=True
    )
    '''
    # Option 2: Uncomment to predict on a single image
    predict_single_image(
        model_path=Config.MINI_MODEL_PATH,
        image_path="C:/Users/ahmed/Dropbox/PC/Pictures/Screenshots/Screenshot 2025-07-04 141651.png"
    )
    '''
