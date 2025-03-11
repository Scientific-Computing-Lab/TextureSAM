import os
import re
import cv2
import numpy as np
import torch
import pandas as pd
from torchmetrics.functional.segmentation import mean_iou

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def unify_instances(instance_masks):
    if not instance_masks:
        return np.zeros((256, 256), dtype=np.int32)

    instance_masks.sort(key=lambda x: np.count_nonzero(x), reverse=True)
    unified_mask = np.zeros_like(instance_masks[0], dtype=np.int32)

    for idx, mask in enumerate(instance_masks, start=1):
        unified_mask[mask > 0] = idx  

    return unified_mask


def match_instances(pred_mask, label_mask):
    unique_labels = np.unique(label_mask)
    matched_pred = np.zeros_like(label_mask, dtype=np.int32)

    for pred_label in np.unique(pred_mask):
        if pred_label == 0:
            continue

        pred_instance_mask = (pred_mask == pred_label)
        best_label = 0
        max_overlap = 0

        for lbl in unique_labels:
            if lbl == 0:
                continue

            lbl_mask = (label_mask == lbl)
            overlap = np.count_nonzero(pred_instance_mask & lbl_mask)

            if overlap > max_overlap:
                max_overlap = overlap
                best_label = lbl

        matched_pred[pred_instance_mask] = best_label

    return matched_pred


def main():
    unified_labels_dir = "./unified_labels"
    pred_root_dir = "./pred_masks"
    output_csv = "./evaluation_results.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_data = []
    unique_identifiers = []
    pred_files = sorted(os.listdir(pred_root_dir), key=natural_sort_key)

    for filename in pred_files:
        match = re.search(r"val_\d+", filename)
        if match:
            identifier = match.group(0)
            if identifier not in unique_identifiers:
                unique_identifiers.append(identifier)

    unique_identifiers.sort(key=natural_sort_key)

    for identifier in unique_identifiers:
        unified_label_file = f"unified_ADE_{identifier}.png"
        unified_label_path = os.path.join(unified_labels_dir, unified_label_file)

        if not os.path.exists(unified_label_path):
            print(f"Skipping {identifier}: No corresponding unified label found.")
            continue

        try:
            pred_masks = [
                cv2.imread(os.path.join(pred_root_dir, f), cv2.IMREAD_GRAYSCALE)
                for f in pred_files if identifier in f
            ]

            if not pred_masks:
                print(f"No masks found for {identifier}")
                continue

            pred_masks = [(mask > 0).astype(np.uint8) for mask in pred_masks]
            unified_pred_mask_np = unify_instances(pred_masks)

            label_array = cv2.imread(unified_label_path, cv2.IMREAD_GRAYSCALE)
            matched_pred_mask_np = match_instances(unified_pred_mask_np, label_array)

            pred_mask = torch.tensor(matched_pred_mask_np, dtype=torch.int64, device=device)
            unified_label = torch.tensor(label_array, dtype=torch.int64, device=device)
            n_classes = len(torch.unique(unified_label))
            if n_classes > 1:
                iou_score = mean_iou(pred_mask.unsqueeze(0), unified_label.unsqueeze(0), num_classes=n_classes)
                print(f"Mean IoU for {identifier}: {iou_score.item()}")
                results_data.append({"Identifier": identifier, "MeanIoU": iou_score.item()})
            else:
                print(f"Skipping {identifier}: Not enough classes for mIoU calculation.")

        except Exception as e:
            print(f"Error processing {identifier}: {e}")
            continue

    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df.sort_values(by="Identifier", key=lambda col: col.str.extract(r'(\d+)')[0].astype(int), inplace=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No evaluation results to save.")

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
