import numpy as np
import pandas as pd
import os
import re
import cv2
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

def load_masks_from_folder(folder_path, image_id=None):
    """
    Loads binary mask images from a folder.
    Assumes filenames follow the pattern: masked_ADE_val_00000001.jpg.
    If image_id is provided, only loads masks for that image ID.
    """
    if image_id is not None:
        image_id = image_id[len("unified_"):]
        pattern = re.compile(fr"{re.escape(image_id)}")
    else:
        pattern = re.compile(r"masked_(ADE_val_\d+)\.jpg")

    masks = []

    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if not match:
            continue

        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read {mask_path}")
            continue

        masks.append(mask > 0)

    return masks


def compute_iou(mask1, mask2):
    """Computes Intersection over Union (IoU)."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def compute_rand_index(mask1, mask2):
    """Computes Adjusted Rand Index (ARI) between two masks."""
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    return adjusted_rand_score(mask1_flat, mask2_flat)


def main():
    gt_folder = "./path_to_ground_truth_masks"
    pred_folder = "./masks_output"
    output_csv = "./evaluation_results.csv"

    results = [] 

    for gt_filename in tqdm(os.listdir(gt_folder), desc="Processing Images"):
        if not gt_filename.endswith(".png"):
            continue

        image_number = os.path.splitext(gt_filename)[0]
        gt_mask_path = os.path.join(gt_folder, gt_filename)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: Could not read {gt_mask_path}")
            continue

        unique_labels = np.unique(gt_mask)
        gt_masks = [(label, gt_mask == label) for label in unique_labels]

        pred_masks = load_masks_from_folder(pred_folder, image_number)

        iou_list = []
        rand_index_list = []

        for label_id, gt_instance_mask in gt_masks:
            for pred_mask in pred_masks:
                if np.logical_and(gt_instance_mask, pred_mask).sum() > 0:
                    iou = compute_iou(gt_instance_mask, pred_mask)
                    rand_index = compute_rand_index(gt_instance_mask, pred_mask)
                    iou_list.append(iou)
                    rand_index_list.append(rand_index)

        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        avg_rand_index = sum(rand_index_list) / len(rand_index_list) if rand_index_list else 0

        print(f"Image: {image_number} | Avg IoU: {avg_iou:.4f} | Avg Rand Index: {avg_rand_index:.4f}")
        results.append((image_number, avg_iou, avg_rand_index))

    # **Save results to CSV once at the end**
    df = pd.DataFrame(results, columns=["Image Number", "Average IoU", "Average Rand Index"])
    df.to_csv(output_csv, index=False)

    # **Compute and display final averages**
    if len(results) > 0:
        mean_iou = df["Average IoU"].mean()
        mean_rand_index = df["Average Rand Index"].mean()
        print(f"\n Overall Mean IoU: {mean_iou:.4f}, Overall Mean Rand Index: {mean_rand_index:.4f}")

    print(" Evaluation complete! Results saved to:", output_csv)

if __name__ == "__main__":
    main()
  