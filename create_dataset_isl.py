"""
Processes images in ./isl_data/<label>/*.jpg or png
to extract MediaPipe hand landmarks for up to TWO hands (Left then Right)
and save to data_isl.pickle.
Also writes labels_isl.txt from folder names.

Usage:
    python create_dataset_isl.py --data_dir ./isl_data --out data_isl.pickle
"""
import os
import pickle
import cv2
import mediapipe as mp
from tqdm import tqdm
import argparse
from utils_isl import load_labels_from_dir, pad_or_truncate, TARGET_VECTOR_LEN, PER_HAND_LEN

def build_two_hand_vector(results):
    """
    results: mediapipe hands.process(...) results
    Returns a length-TARGET_VECTOR_LEN vector where first half = Left hand (if present),
    second half = Right hand (if present). Missing hand => zeros in its slot.
    We use landmark.x/y minus min(x),min(y) per hand for some normalization.
    """
    # zeros default
    vec = [0.0] * TARGET_VECTOR_LEN
    if not results.multi_hand_landmarks:
        return vec
    # gather handedness labels (parallel to multi_hand_landmarks)
    handedness = []
    if results.multi_handedness:
        for h in results.multi_handedness:
            # label may be 'Left' or 'Right'
            try:
                handedness.append(h.classification[0].label)
            except Exception:
                handedness.append('Unknown')
    # Build per-hand dict by handedness using first occurrence
    hand_map = {}  # 'Left' -> landmarks, 'Right' -> landmarks
    for idx, lm in enumerate(results.multi_hand_landmarks):
        label = handedness[idx] if idx < len(handedness) else f'hand{idx}'
        x_ = [p.x for p in lm.landmark]
        y_ = [p.y for p in lm.landmark]
        x_min, y_min = min(x_), min(y_)
        coords = []
        for i in range(len(lm.landmark)):
            coords.append(lm.landmark[i].x - x_min)
            coords.append(lm.landmark[i].y - y_min)
        # pad/truncate per hand to PER_HAND_LEN
        coords = pad_or_truncate(coords, PER_HAND_LEN).tolist()
        # store
        if label not in hand_map:
            hand_map[label] = coords
    # compose final vec: Left then Right
    left = hand_map.get('Left', [0.0]*PER_HAND_LEN)
    right = hand_map.get('Right', [0.0]*PER_HAND_LEN)
    return left + right

def process_hand_landmarks(data_dir, output_file='data_isl.pickle',
                           min_confidence=0.3, max_hands=2):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           min_detection_confidence=min_confidence,
                           max_num_hands=max_hands)
    data = []
    labels = []
    errors = []

    # derive labels
    labels_list = load_labels_from_dir(data_dir, out_labels_file='labels_isl.txt')

    image_paths = []
    for label in labels_list:
        folder = os.path.join(data_dir, label)
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append((os.path.join(folder, fn), label))

    for img_path, label in tqdm(image_paths, desc="Processing images", unit="img"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                errors.append(f"Unreadable: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            vec = build_two_hand_vector(results)
            # if vec is all zeros (no hand detected), record error
            if all(v == 0.0 for v in vec):
                errors.append(f"No hands detected: {img_path}")
                continue
            data.append(vec)
            labels.append(label)
        except Exception as e:
            errors.append(f"Error {img_path}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Saved {len(data)} samples to {output_file}")
    print(f"Errors: {len(errors)} (see processing_errors.log)")
    if errors:
        with open('processing_errors.log', 'w') as el:
            el.write('\n'.join(errors))

    return data, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./isl_data', help='Folder with subfolders per class')
    parser.add_argument('--out', default='data_isl.pickle')
    args = parser.parse_args()
    process_hand_landmarks(args.data_dir, args.out)
