import os
import re
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchmetrics.functional.segmentation import mean_iou


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_masks_from_folder(folder_path):
    """
    Loads binary mask images from a folder.
    Assumes filenames follow the pattern: mask_{maskID}_{imageID}.png.
    Returns a dictionary mapping image IDs to lists of boolean mask arrays.
    """
    masks_by_image_id = {}
    pattern = re.compile(r"mask_\d+_(\d+)\.png")  

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            image_id = match.group(1)   

            mask_path = os.path.join(folder_path, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Could not read {mask_path}")
                continue

            # Store masks using image ID as the key
            if image_id not in masks_by_image_id:
                masks_by_image_id[image_id] = []
            masks_by_image_id[image_id].append(mask == 255)  # Convert to boolean mask

    return masks_by_image_id


def match_instance_labels(pred_masks, label):
    """Match predicted mask instances to ground truth labels based on pixel overlap."""
    matched_masks = []
    
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, dtype=torch.int64)

    label_unique = torch.unique(label)
    for pred_mask in pred_masks:
        pred_segmentation = pred_mask['segmentation']
        
        if isinstance(pred_segmentation, np.ndarray):
            pred_segmentation = torch.tensor(pred_segmentation, dtype=torch.int64)
        
        best_label = 0
        max_overlap = 0

        # Find the label in the ground truth that overlaps the most with the predicted mask
        for lbl in label_unique:
            lbl_mask = (label == lbl)
            overlap = torch.count_nonzero(pred_segmentation & lbl_mask).item()
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = lbl.item()

        matched_mask = torch.zeros_like(pred_segmentation, dtype=torch.int64)
        matched_mask[pred_segmentation.bool()] = best_label

        matched_masks.append(matched_mask.numpy())

    return sorted(matched_masks, key=lambda x: np.count_nonzero(x), reverse=True)


def main():
    label_dir = "./ground_truth_labels"
    pred_masks_dir = "./masks_output"
    output_csv_path = "./evaluation_results.csv"
    rwtd_dataset = False 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pred_masks_by_image = load_masks_from_folder(pred_masks_dir)
    
    results_data = []
    for filename in sorted(os.listdir(label_dir), key=natural_sort_key):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_num_str = os.path.splitext(filename)[0]
            try:
                image_num = int(image_num_str)
            except ValueError:
                print(f"Skipping {filename}: filename does not represent an image number.")
                continue

            label_path = os.path.join(label_dir, filename)
            label = np.array(Image.open(label_path).convert("L"))  
            label_tensor = torch.tensor(label, dtype=torch.int64, device=device)
            n_classes = len(torch.unique(label_tensor))

            if rwtd_dataset:
                label_inverse = 255 - label
                label_tensor_inverse = torch.tensor(label_inverse, dtype=torch.int64, device=device)
            
            if image_num_str not in pred_masks_by_image:
                print(f"No predicted masks found for image number {image_num} corresponding to file {filename}.")
                continue

            pred_masks_list = [{'segmentation': mask.astype(np.uint8)} for mask in pred_masks_by_image[image_num_str]]

            matched_masks = match_instance_labels(pred_masks_list, label)
            
            if rwtd_dataset:
                matched_masks_inverse = match_instance_labels(pred_masks_list, label_inverse)

            if not matched_masks:
                print(f"No matched masks for {filename}.")
                continue

            pred_label_tensor = torch.zeros_like(label_tensor, device=device)
            for mask in matched_masks:
                mask_tensor = torch.tensor(mask, dtype=torch.int64, device=device)
                pred_label_tensor[mask_tensor > 0] = mask_tensor[mask_tensor > 0]

            if rwtd_dataset:
                pred_label_tensor_inverse = torch.zeros_like(label_tensor_inverse, device=device)
                for mask in matched_masks_inverse:
                    mask_tensor_inv = torch.tensor(mask, dtype=torch.int64, device=device)
                    pred_label_tensor_inverse[mask_tensor_inv > 0] = mask_tensor_inv[mask_tensor_inv > 0]

            try:
                iou_score = mean_iou(pred_label_tensor.unsqueeze(0), label_tensor.unsqueeze(0), num_classes=n_classes)
                if rwtd_dataset:
                    iou_score_inverse = mean_iou(pred_label_tensor_inverse.unsqueeze(0), label_tensor_inverse.unsqueeze(0), num_classes=n_classes)
                    mean_iou_score = (iou_score.item() + iou_score_inverse.item()) / 2
                    max_iou_score = max(iou_score.item(), iou_score_inverse.item())
                else:
                    mean_iou_score = iou_score.item()
                    max_iou_score = iou_score.item()

                print(f"Mean IoU score for {filename}: {max_iou_score}")
                results_data.append({"Filename": filename, "MeanIoU": mean_iou_score, "MaxIoU": max_iou_score})
            except Exception as e:
                print(f"Error calculating IoU for {filename}: {e}")

    results_df = pd.DataFrame(results_data)
    if not results_df.empty:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
    else:
        print("No evaluation results to save.")

    print("Evaluation complete!")

if __name__ == "__main__":
    main()
