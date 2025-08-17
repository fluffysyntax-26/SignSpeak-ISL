# ISL Static Sign Language Recognition

## Overview
This project provides a complete pipeline for recognizing **Indian Sign Language (ISL) static signs** using computer vision and machine learning. It includes scripts for:
1. Collecting sign images via webcam.
2. Processing images to extract hand landmarks using MediaPipe.
3. Training a RandomForest classifier for sign recognition.
4. Running a **Streamlit** web application for real-time and image-based recognition.

The system supports **one-hand and two-hand static gestures** and is modular, allowing for the addition of new signs or changes to the dataset.

---

## Project Structure

```
.
├── collect_images.py          # Capture ISL sign images via webcam
├── create_dataset_isl.py      # Extract MediaPipe landmarks and save dataset
├── train_classifier_isl.py    # Train classifier and save model
├── recogniser_streamlit.py    # Streamlit app for sign recognition
├── utils_isl.py               # Utility functions and constants
├── isl_data/                  # Dataset folder (created after image collection)
├── labels_isl.txt             # Class labels file (auto-generated)
├── data_isl.pickle            # Processed dataset (generated)
├── model_isl.p                # Trained classifier model (generated)
└── confusion_matrix_isl.png   # Confusion matrix plot (generated)
```

---

## Requirements

Install the dependencies before running any script:

```bash
pip install opencv-python mediapipe scikit-learn streamlit joblib pillow tqdm matplotlib numpy
```

---

## Workflow

### 1. Collect ISL Sign Images
Use your webcam to capture images for each sign gesture.

```bash
python collect_images.py
```
- Default signs are defined in `ISL_SIGNS` in `collect_images.py`.
- Each sign will have **1000 images** captured by default (can be changed in `DATASET_SIZE`).
- Images are stored in `./isl_data/<SIGN>/`.

---

### 2. Create the Dataset
Process the collected images to extract MediaPipe hand landmarks.

```bash
python create_dataset_isl.py --data_dir ./isl_data --out data_isl.pickle
```
- Saves the processed feature vectors and labels in `data_isl.pickle`.
- Saves class labels in `labels_isl.txt`.
- Logs unreadable or undetected-hand images in `processing_errors.log`.

---

### 3. Train the Classifier
Train a **RandomForest** classifier using the processed dataset.

```bash
python train_classifier_isl.py --data data_isl.pickle --out model_isl.p
```
- Outputs a trained model (`model_isl.p`) and a `confusion_matrix_isl.png`.
- Also prints accuracy, classification report, and cross-validation scores.

---

### 4. Run the Recognition App
Launch the **Streamlit** application for real-time or image-based recognition.

```bash
streamlit run recogniser_streamlit.py
```

Features:
- Upload an image for classification.
- Enable webcam for live predictions with smoothing and frame-skipping options.
- Supports both one-hand and two-hand static signs.

---

## Utility Functions
The `utils_isl.py` script provides:
- `ensure_dir`: Creates directories if not present.
- `load_labels_from_dir`: Loads labels from a dataset directory and writes them to a file.
- `pad_or_truncate`: Adjusts landmark vectors to a fixed length.
- Predefined constants for vector lengths based on hand landmarks.

---

## Notes
- The pipeline is designed for **static ISL signs** only.
- Ensure proper lighting and background contrast for optimal detection.
- Model performance depends heavily on dataset size and quality.
- The trained model and scaler are stored together in `model_isl.p`.

---

## Example Usage Flow

```bash
# Step 1: Collect images
python collect_images.py

# Step 2: Create dataset
python create_dataset_isl.py --data_dir ./isl_data --out data_isl.pickle

# Step 3: Train classifier
python train_classifier_isl.py --data data_isl.pickle --out model_isl.p

# Step 4: Run recognition app
streamlit run recogniser_streamlit.py
```

---

## License
This project is released under the MIT License. You are free to use, modify, and distribute it, provided the original authors are credited.
