import os
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_labels_from_dir(data_dir, out_labels_file='labels_isl.txt'):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    with open(out_labels_file, 'w') as f:
        for l in labels:
            f.write(f"{l}\n")
    return labels

def pad_or_truncate(vector, target_len):
    """Ensure vector has length target_len: pad with zeros or truncate."""
    v = list(vector)
    if len(v) > target_len:
        return np.asarray(v[:target_len], dtype=np.float32)
    elif len(v) < target_len:
        pad = [0.0] * (target_len - len(v))
        return np.asarray(v + pad, dtype=np.float32)
    else:
        return np.asarray(v, dtype=np.float32)

# constants: 21 landmarks per hand, (x,y) only => 21*2 = 42 per hand
LANMARKS_PER_HAND = 21
COORDS_PER_LM = 2
PER_HAND_LEN = LANMARKS_PER_HAND * COORDS_PER_LM

# we will store two hands: Left then Right -> total vector length:
TWO_HAND_VECTOR_LEN = PER_HAND_LEN * 2
TARGET_VECTOR_LEN = TWO_HAND_VECTOR_LEN
