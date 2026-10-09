"""
evaluate.py - LeafHealth Detector model evaluation and diagnostics.
Run: python evaluate.py
Output: static/eval_results.json
"""

import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import CNN

MODEL_PATH = "plant_disease_model_1_latest.pt"
OUTPUT_JSON = "static/eval_results.json"
IMAGE_SIZE = (224, 224)

# This checkpoint was trained on RGB images converted with ToTensor() only.
# Applying ImageNet normalization collapses predictions to a few classes.
NORMALIZE_INPUT = False

IMAGES_USED = 0
random.seed(42)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

IDX_TO_CLASS = CNN.idx_to_classes
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}
NUM_CLASSES = len(IDX_TO_CLASS)

print("Loading model...")
print(f"Model path: {os.path.abspath(MODEL_PATH)}")
print(f"Preprocessing: RGB resize={IMAGE_SIZE}, to_tensor, normalize={NORMALIZE_INPUT}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN.CNN(NUM_CLASSES)
state_dict = torch.load(MODEL_PATH, map_location=device)
load_result = model.load_state_dict(state_dict, strict=True)
model.eval()
model.to(device)
print(f"Model loaded on {device} | Classes: {NUM_CLASSES}")
print(f"Missing keys: {load_result.missing_keys}")
print(f"Unexpected keys: {load_result.unexpected_keys}")
print(f"Classifier output layer: {tuple(model.dense_layers[4].weight.shape)}")
print("Class mapping:")
for idx in range(NUM_CLASSES):
    print(f"  {idx:02d}: {IDX_TO_CLASS[idx]}")


def transform_image(image_path):
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    input_data = TF.to_tensor(image)
    if NORMALIZE_INPUT:
        input_data = TF.normalize(
            input_data,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    return input_data.unsqueeze(0).to(device)


def predict(image_path):
    try:
        input_data = transform_image(image_path)
        with torch.no_grad():
            output = model(input_data)
            probs = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        return pred_idx.item(), conf.item()
    except Exception as exc:
        print(f"Skipped {image_path}: {exc}")
        return None, None


def find_dataset():
    for root in ["dataset", "data", "PlantVillage", "plant_village", "test", "valid", "../dataset"]:
        if os.path.exists(root):
            for current_root, dirs, _ in os.walk(root):
                matched = sum(1 for class_name in CLASS_TO_IDX if class_name in dirs)
                if matched >= 10:
                    print(f"Dataset found: {current_root}")
                    return current_root
    return None


def resolve_class_folder(dataset_root, class_name):
    class_folder = os.path.join(dataset_root, class_name)
    if os.path.exists(class_folder):
        return class_folder

    for current_root, dirs, _ in os.walk(dataset_root):
        for dirname in dirs:
            if class_name.lower() == dirname.lower():
                return os.path.join(current_root, dirname)
    return None


def evaluate_on_dataset(dataset_root, max_per_class=50):
    global IMAGES_USED
    y_true, y_pred = [], []
    class_counts = {}
    sample_predictions = []

    for class_name, true_idx in CLASS_TO_IDX.items():
        class_folder = resolve_class_folder(dataset_root, class_name)
        if not class_folder:
            print(f"  {true_idx:02d} {class_name[:50]:<50} MISSING")
            continue

        images = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[class_name] = len(images)
        sample = random.sample(images, min(max_per_class, len(images)))
        correct = 0

        for img_name in sample:
            pred_idx, conf = predict(os.path.join(class_folder, img_name))
            if pred_idx is None:
                continue

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            IMAGES_USED += 1
            if pred_idx == true_idx:
                correct += 1
            if len(sample_predictions) < 20:
                sample_predictions.append({
                    "file": img_name,
                    "true_index": true_idx,
                    "true_class": class_name,
                    "pred_index": pred_idx,
                    "pred_class": IDX_TO_CLASS[pred_idx],
                    "confidence": round(float(conf) * 100, 2),
                })

        print(f"  {true_idx:02d} {class_name[:50]:<50} {correct}/{len(sample)} total_in_folder={len(images)}")

    return y_true, y_pred, class_counts, sample_predictions


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    per_class_p, per_class_r, per_class_f1, per_class_support = [], [], [], []

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        conf_mat[true_idx][pred_idx] += 1

    for c in range(NUM_CLASSES):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = int(np.sum(y_true == c))
        per_class_p.append(p)
        per_class_r.append(r)
        per_class_f1.append(f)
        per_class_support.append(support)

    misclasses = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and conf_mat[i][j] > 0:
                misclasses.append({
                    "true": IDX_TO_CLASS[i],
                    "predicted": IDX_TO_CLASS[j],
                    "count": int(conf_mat[i][j]),
                })
    misclasses.sort(key=lambda x: x["count"], reverse=True)

    pred_counts = Counter(int(p) for p in y_pred)
    per_class_accuracy = {
        IDX_TO_CLASS[i]: round((conf_mat[i][i] / conf_mat[i].sum()) * 100, 2)
        if conf_mat[i].sum() else 0.0
        for i in range(NUM_CLASSES)
    }
    classification_report = {
        IDX_TO_CLASS[i]: {
            "precision": round(per_class_p[i] * 100, 2),
            "recall": round(per_class_r[i] * 100, 2),
            "f1_score": round(per_class_f1[i] * 100, 2),
            "support": per_class_support[i],
        }
        for i in range(NUM_CLASSES)
    }
    weighted_precision = sum(
        per_class_p[i] * per_class_support[i] for i in range(NUM_CLASSES)
    ) / len(y_true)
    weighted_recall = sum(
        per_class_r[i] * per_class_support[i] for i in range(NUM_CLASSES)
    ) / len(y_true)
    weighted_f1 = sum(
        per_class_f1[i] * per_class_support[i] for i in range(NUM_CLASSES)
    ) / len(y_true)
    classification_report["accuracy"] = {
        "precision": None,
        "recall": None,
        "f1_score": round(accuracy * 100, 2),
        "support": int(len(y_true)),
    }
    classification_report["macro_avg"] = {
        "precision": round(float(np.mean(per_class_p)) * 100, 2),
        "recall": round(float(np.mean(per_class_r)) * 100, 2),
        "f1_score": round(float(np.mean(per_class_f1)) * 100, 2),
        "support": int(len(y_true)),
    }
    classification_report["weighted_avg"] = {
        "precision": round(float(weighted_precision) * 100, 2),
        "recall": round(float(weighted_recall) * 100, 2),
        "f1_score": round(float(weighted_f1) * 100, 2),
        "support": int(len(y_true)),
    }

    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(float(np.mean(per_class_f1)) * 100, 2),
        "macro_precision": round(float(np.mean(per_class_p)) * 100, 2),
        "macro_recall": round(float(np.mean(per_class_r)) * 100, 2),
        "model_path": os.path.abspath(MODEL_PATH),
        "preprocessing": {
            "resize": list(IMAGE_SIZE),
            "rgb": True,
            "to_tensor": True,
            "normalize": NORMALIZE_INPUT,
            "mean": None,
            "std": None,
        },
        "class_mapping": {str(i): IDX_TO_CLASS[i] for i in range(NUM_CLASSES)},
        "top_predicted_classes": [
            {"index": idx, "class": IDX_TO_CLASS[idx], "count": count}
            for idx, count in pred_counts.most_common(15)
        ],
        "per_class_accuracy": per_class_accuracy,
        "per_class_f1": {IDX_TO_CLASS[i]: round(per_class_f1[i] * 100, 2) for i in range(NUM_CLASSES)},
        "classification_report": classification_report,
        "confusion_matrix": conf_mat.tolist(),
        "top_misclassifications": misclasses[:15],
        "images_evaluated": IMAGES_USED,
        "real_evaluation": True,
    }


def main():
    print("=" * 60)
    print("  LeafHealth - REAL Model Evaluation")
    print("=" * 60)

    dataset_root = find_dataset()

    if dataset_root is None:
        print("\nDataset folder nahi mila!")
        print("Manual path enter karo (ya Enter dabao skip ke liye):")
        manual = input("Path: ").strip()
        if manual and os.path.exists(manual):
            dataset_root = manual
        else:
            print("\nPlantVillage dataset download karo:")
            print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
            print("Extract karke 'dataset/' folder mein rakho, phir dobara run karo.")
            return

    y_true, y_pred, class_counts, sample_predictions = evaluate_on_dataset(dataset_root, max_per_class=50)

    if len(y_true) < 10:
        print("Bahut kam images mili! Dataset folder structure check karo.")
        return

    print(f"\nTotal images: {len(y_true)}")
    metrics = calculate_metrics(y_true, y_pred)
    metrics["dataset_root"] = os.path.abspath(dataset_root)
    metrics["dataset_class_counts"] = class_counts
    metrics["sample_predictions"] = sample_predictions

    os.makedirs("static", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  REAL RESULTS ({metrics['images_evaluated']} images)")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  F1 Score  : {metrics['macro_f1']}%")
    print(f"  Precision : {metrics['macro_precision']}%")
    print(f"  Recall    : {metrics['macro_recall']}%")
    print("=" * 60)
    print("\nTop predicted classes:")
    for item in metrics["top_predicted_classes"]:
        print(f"  {item['index']:02d} {item['class']:<55} {item['count']}")
    print("\nFirst 20 predictions vs actual labels:")
    for item in sample_predictions:
        print(
            f"  true={item['true_index']:02d}:{item['true_class']} | "
            f"pred={item['pred_index']:02d}:{item['pred_class']} | "
            f"conf={item['confidence']}%"
        )
    print("\nConfusion matrix (rows=true, columns=pred):")
    print(np.array(metrics["confusion_matrix"]))
    print("\nPer-class accuracy:")
    for class_name, acc in metrics["per_class_accuracy"].items():
        print(f"  {class_name:<55} {acc:6.2f}%")
    print("\nClassification report:")
    print(f"  {'class':<55} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}")
    for class_name, row in metrics["classification_report"].items():
        precision = "-" if row["precision"] is None else f"{row['precision']:.2f}"
        recall = "-" if row["recall"] is None else f"{row['recall']:.2f}"
        print(
            f"  {class_name:<55} {precision:>9} "
            f"{recall:>9} {row['f1_score']:>9.2f} {row['support']:>9}"
        )
    print(f"Saved: {OUTPUT_JSON}")
    print("Ab /evaluation refresh karo - real numbers dikhenge!")


if __name__ == "__main__":
    main()
