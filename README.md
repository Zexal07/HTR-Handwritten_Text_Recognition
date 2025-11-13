# HTR-Handwritten_Text_Recognition

This project implements a complete pipeline for Handwritten Text Recognition (HTR) using a Convolutional Recurrent Neural Network (CRNN) model in PyTorch. The model is trained to extract single-line text from images, using the "Handwritten Name Recognition" dataset from Kaggle.



The repository includes scripts for:
1.  **Initial Training:** Training the model from scratch on a subset of data.
2.  **Continue Training:** Loading a saved checkpoint and continuing training.
3.  **Inference:** Using the trained model to predict text from new images (both single and batch).

## Features

* **Model:** A robust CRNN (CNN + Bi-LSTM) architecture.
* **CNN Backbone:** A deep convolutional network with `BatchNorm` for stable and efficient feature extraction.
* **RNN Sequence:** A 3-layer bidirectional LSTM with `dropout` to capture contextual information in the text.
* **Loss Function:** `nn.CTCLoss`, which is ideal for sequence-to-sequence tasks where the input-output alignment is unknown.
* **Training:**
    * Gradient clipping to prevent exploding gradients.
    * `Adam` optimizer.
    * `ReduceLROnPlateau` learning rate scheduler to adjust LR based on loss.
    * Data augmentation (RandomAffine, ColorJitter) to improve model generalization.
* **Inference:**
    * Greedy CTC decoder to translate model output into human-readable text.
    * Calculates a confidence score for each prediction.
    * Scripts for both single-image and batch-directory prediction.
    * Saves batch results to `predictions.csv` and `predictions.txt`.

## Dataset

This model is designed to be trained on the **Handwriting Recognition** dataset from Kaggle.

1.  **Download:** [Download the dataset here](https://www.kaggle.com/datasets/landlord/handwriting-recognition).
2.  **Structure:** Unzip the files and organize them into the following directory structure in your project's root folder:
   ```bash
HTR-Handwritten_Text_Recognition/
├── train.py
├── continue_train.py
├── predict.py
├── CSV/
│ ├── written_name_train.csv
│ └── written_name_test.csv
| └── written_name_validation.csv
├── train_v2/
│ └── train/
│ ├── TRAIN_1.jpg
│ ├── TRAIN_2.jpg
│ └── ...
└── test_v2/
└── test/
├── TEST_1.jpg
├── TEST_2.jpg
└── ...
└── validation_v2/
└── validation/
├── VALIDATION_1.jpg
├── VALIDATION_2.jpg
└── ...
```
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/HTR-Handwritten_Text_Recognition.git](https://github.com/your-username/HTR-Handwritten_Text_Recognition.git)
    cd HTR-Handwritten_Text_Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file with the following content:
    ```txt
    torch
    torchvision
    pandas
    Pillow
    tqdm
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

The provided code is split into three main files. You should save them as `train.py`, `continue_train.py`, and `predict.py`.

### 1. Initial Training

This script trains the model from scratch for 10 epochs (by default) on a subset of 5000 random samples. It will save the model with the best validation loss to `htr_crnn_mini.pth`.

```bash
python train.py
```
### 2. Continue Training
After running the initial training, you can use this script to load the saved htr_crnn_mini.pth and continue training for 50 additional epochs.

```bash
python continue_train.py
```
### 3. Inference (Prediction)
This script uses the trained htr_crnn_mini.pth to perform predictions.

A. Batch Prediction (Default)

By default, the script will:

Run predictions on all images in the test_v2/test/ directory.

Save the results to an output/ folder (predictions.csv and predictions.txt).

```bash
python predict.py
```
B. Single Image Prediction

To predict a single image, you need to edit the predict.py file:

Comment out the "Batch prediction" block.

Uncomment the "Single image" block.

Change the path to your desired image.

Python
```bash
if __name__ == "__main__":
    print("=" * 60)
    print("CRNN MODEL - TESTING/INFERENCE")
    print("=" * 60 + "\n")
    
    '''
    # Option 1: Batch prediction on test directory (COMMENT THIS OUT)
    img_dir = Config.TEST_IMG_DIR
    output_folder = "output"
    
    predict_batch(
        model_path=Config.MINI_MODEL_PATH,
        image_dir=img_dir,
        output_dir=output_folder,
        save_csv=True
    )
    '''
    
    # Option 2: Uncomment to predict on a single image (UNCOMMENT THIS)
    predict_single_image(
        model_path=Config.MINI_MODEL_PATH,
        image_path="path/to/your/image.png"  # <-- CHANGE THIS PATH
    )
  ```  
Then, run the script:

```bash
python predict.py
```
License
This project is not licensed.
