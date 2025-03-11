import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import cv2
from sklearn.metrics import adjusted_rand_score

def load_masks_from_folder(folder_path):
    """
    Loads binary mask images from a folder.
    Assumes filenames follow the pattern: mask_{imageNumber}_{maskID}.png.
    Returns a dictionary mapping mask IDs to boolean mask arrays.
    """
    masks_by_id = {}
    pattern = re.compile(r"mask_(\d+)_(\d+)\.png") 

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            mask_id = int(match.group(2))  
            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue

            if mask_id not in masks_by_id:
                masks_by_id[mask_id] = []
            masks_by_id[mask_id].append(mask == 255)

    return masks_by_id


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def compute_rand_index(mask1, mask2):
    """
    Computes the Adjusted Rand Index (ARI) between two masks.
    Flattens both masks and computes ARI using sklearn.
    """
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    
    return adjusted_rand_score(mask1_flat, mask2_flat)


def calculate_metrics_per_gt(masks_by_image, labels_by_image):
    """
    Computes the IoU and Rand Index between each ground truth mask and all predicted masks that overlap with it.
    
    Returns:
        - Dictionary containing average IoU and Rand Index per GT mask.
        - Overall average IoU and Rand Index across all GT masks.
    """
    results = {}
    total_iou = 0
    total_rand_index = 0
    count = 0

    for image_number, gt_masks in labels_by_image.items():
        image_number = int(image_number) 
        
        if image_number not in masks_by_image:
            print(f"No predicted masks found for image {image_number}")
            continue

        pred_masks = masks_by_image[image_number]  

        for gt_id, gt_mask in gt_masks:
            iou_list = []
            rand_index_list = []

            for pred_mask in pred_masks:
                if np.logical_and(gt_mask, pred_mask).sum() > 0: 
                    iou = compute_iou(gt_mask, pred_mask)
                    rand_index = compute_rand_index(gt_mask, pred_mask)

                    iou_list.append(iou)
                    rand_index_list.append(rand_index)

            if iou_list:  
                avg_iou = sum(iou_list) / len(iou_list)
                avg_rand_index = sum(rand_index_list) / len(rand_index_list)
                total_iou += avg_iou
                total_rand_index += avg_rand_index
                count += 1
            else:
                avg_iou = 0  
                avg_rand_index = 0

            results[(image_number, gt_id)] = {
                "average_iou": avg_iou,
                "average_rand_index": avg_rand_index,
                "all_iou": iou_list,
                "all_rand_index": rand_index_list
            }

    overall_average_iou = total_iou / count if count > 0 else 0
    overall_average_rand_index = total_rand_index / count if count > 0 else 0
    return results, overall_average_iou, overall_average_rand_index




def load_gt_masks_as_instances(folder_path):
    """
    Loads GT masks where each label represents a different instance.
    Extracts each unique label as a separate boolean mask.

    Returns:
        Dictionary mapping image filenames (without extension) to a list of (label_id, boolean_mask).
    """
    gt_masks_by_image = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"): 
            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue

            unique_labels = np.unique(mask)
            image_name = os.path.splitext(filename)[0]
            gt_masks_by_image[image_name] = []

            for label in unique_labels:
                instance_mask = mask == label 
                gt_masks_by_image[image_name].append((label, instance_mask))

    return gt_masks_by_image




def main():
    gt_folder = "./path_to_ground_truth_masks"
    pred_folder = "./masks_output"
    output_csv = "./evaluation_results.csv"

    gt_masks_by_image = load_gt_masks_as_instances(gt_folder)
    pred_masks_by_image = load_masks_from_folder(pred_folder)
    
    metrics_results, overall_avg_iou, overall_avg_rand_index = calculate_metrics_per_gt(pred_masks_by_image, gt_masks_by_image)
    
    metrics_data = []
    for (image_num, gt_id), data in metrics_results.items():
        row = {
            "Image Number": image_num,
            "GT Mask ID": gt_id,
            "Average IoU": data["average_iou"],
            "Average Rand Index": data["average_rand_index"],
            "All IoU Scores": "; ".join(map(str, data["all_iou"])),  
            "All Rand Index Scores": "; ".join(map(str, data["all_rand_index"]))
        }
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(output_csv, index=False)
    
    print(f"Metrics results saved to {output_csv}")
    print(f"Overall Average IoU: {overall_avg_iou:.4f}")
    print(f"Overall Average Rand Index: {overall_avg_rand_index:.4f}")

if __name__ == "__main__":
    main()
