import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from typing import Dict, Tuple


class Config:
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-?!&"
    VOCAB_SIZE = len(CHARS) + 1
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    HIDDEN_SIZE = 256
    MODEL_PATH = "htr_crnn_mini.pth"
    SUPPORTED_FORMATS = (
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
        ("All files", "*.*")
    )


class CRNN(nn.Module):
    def __init__(self, img_height, vocab_size, hidden_size):
        super(CRNN, self).__init__()
        
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
        
        self.map_to_rnn = nn.Linear(512, hidden_size)
        
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
        cnn_out = cnn_out.squeeze(2).permute(0, 2, 1)
        rnn_input = self.map_to_rnn(cnn_out).permute(1, 0, 2)
        rnn_out, _ = self.rnn(rnn_input)
        output = self.linear(rnn_out)
        return nn.functional.log_softmax(output, dim=2)


class HandwritingRecognizer:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else 
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.char_to_int, self.int_to_char = self._create_mappings(Config.CHARS)
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @staticmethod
    def _create_mappings(chars: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        char_to_int = {char: i + 1 for i, char in enumerate(chars)}
        int_to_char = {i + 1: char for i, char in enumerate(chars)}
        char_to_int['CTC_BLANK'] = 0
        int_to_char[0] = ''
        return char_to_int, int_to_char
    
    def _load_model(self, model_path: str) -> nn.Module:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = CRNN(Config.IMG_HEIGHT, Config.VOCAB_SIZE, Config.HIDDEN_SIZE)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _decode_ctc(self, output: torch.Tensor) -> Tuple[str, float]:
        probs = output.exp()
        preds = output.argmax(dim=1)
        max_probs = probs.max(dim=1)[0]
        
        decoded_text = []
        confidence_scores = []
        prev_idx = -1
        
        for i, char_idx in enumerate(preds.cpu().numpy()):
            if char_idx != 0 and char_idx != prev_idx:
                decoded_text.append(self.int_to_char.get(int(char_idx), '?'))
                confidence_scores.append(max_probs[i].item())
            prev_idx = char_idx
        
        avg_confidence = (sum(confidence_scores) / len(confidence_scores) 
                         if confidence_scores else 0.0)
        return "".join(decoded_text), avg_confidence
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor).squeeze(1)
            text, confidence = self._decode_ctc(output)
        
        return text, confidence


class HTRApp:
    def __init__(self, root, recognizer):
        self.root = root
        self.recognizer = recognizer
        self.current_image_path = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        self.root.title("Handwriting Text Recognition")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        tk.Button(
            button_frame,
            text="Select Image",
            command=self._select_image,
            font=("Arial", 12),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame,
            text="Extract Text",
            command=self._extract_text,
            font=("Arial", 12),
            padx=20,
            pady=10
        ).pack(side=tk.LEFT, padx=5)
        
        image_frame = tk.LabelFrame(main_frame, text="Selected Image", font=("Arial", 10))
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.image_label = tk.Label(image_frame, text="No image selected", bg="lightgray")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_frame = tk.LabelFrame(main_frame, text="Extracted Text", font=("Arial", 10))
        result_frame.pack(fill=tk.BOTH, pady=(0, 10))
        
        self.result_text = tk.Text(result_frame, height=4, font=("Arial", 11), wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, padx=10, pady=10)
        
        self.confidence_label = tk.Label(
            main_frame,
            text="Confidence: N/A",
            font=("Arial", 10),
            anchor=tk.W
        )
        self.confidence_label.pack(fill=tk.X)
    
    def _select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Handwritten Image",
            filetypes=Config.SUPPORTED_FORMATS
        )
        
        if not file_path:
            return
        
        try:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.result_text.delete(1.0, tk.END)
            self.confidence_label.config(text="Confidence: N/A")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _display_image(self, image_path):
        image = Image.open(image_path)
        
        display_width = 760
        display_height = 300
        image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def _extract_text(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first")
            return
        
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, "Processing...")
            self.root.update()
            
            text, confidence = self.recognizer.predict(self.current_image_path)
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, text if text else "(No text detected)")
            self.confidence_label.config(text=f"Confidence: {confidence:.4f}")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, "Error during extraction")
            messagebox.showerror("Extraction Error", str(e))


def main():
    try:
        print("Loading model...")
        recognizer = HandwritingRecognizer(Config.MODEL_PATH)
        print(f"Model loaded successfully on {recognizer.device}")
        
        root = tk.Tk()
        app = HTRApp(root, recognizer)
        root.mainloop()
        
    except FileNotFoundError as e:
        messagebox.showerror("Model Error", str(e))
    except Exception as e:
        messagebox.showerror("Initialization Error", str(e))


if __name__ == "__main__":
    main()
