"""
Train a RandomForest classifier on data_isl.pickle and save model_isl.p (joblib)
Usage:
    python train_classifier_isl.py --data data_isl.pickle --out model_isl.p
"""
import pickle
import numpy as np  # ✅ moved here so np is always available
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from joblib import dump
from utils_isl import TARGET_VECTOR_LEN

def train_hand_sign_classifier(data_path='data_isl.pickle', save_path='model_isl.p'):
    with open(data_path, 'rb') as f:
        D = pickle.load(f)
    X = np.asarray(D['data'])
    y = np.asarray(D['labels'])

    # Safety: ensure correct vector length (pad/truncate rows if necessary)
    if X.shape[1] != TARGET_VECTOR_LEN:
        print(f"Fixing vector lengths: expected {TARGET_VECTOR_LEN}, got {X.shape[1]}")
        X2 = []
        for row in X:
            r = list(row)
            if len(r) < TARGET_VECTOR_LEN:
                r = r + [0.0] * (TARGET_VECTOR_LEN - len(r))
            elif len(r) > TARGET_VECTOR_LEN:
                r = r[:TARGET_VECTOR_LEN]
            X2.append(r)
        X = np.asarray(X2, dtype=np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Save model + scaler
    dump({'model': model, 'scaler': scaler}, save_path)
    print(f"Saved model to {save_path}")

    # Confusion matrix image
    classes = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix_isl.png')
    plt.close()

    # Cross-validation scores
    scores = cross_val_score(model, Xs, y, cv=5, n_jobs=-1)
    print("CV scores:", scores)
    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data_isl.pickle')
    parser.add_argument('--out', default='model_isl.p')
    args = parser.parse_args()
    train_hand_sign_classifier(args.data, args.out)
