import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
import json
from torch import randint
import re
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def generate_masks(mask_generator, image_array):
    """Generate masks for a given image array using the mask generator."""
    return mask_generator.generate(image_array)

def show_anns(anns, ax=None, borders=True):
    """Display annotations on an image."""
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def process_images(image_dir, mask_generator, output_dir):
    """Process all images in a directory, generate masks, and save the output."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask_output_dir = os.path.join(output_dir, "masks_output")
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    print(f"Output directory: {output_dir}")
    results = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    for idx, filename in enumerate(tqdm(image_files, desc="Processing Images")):

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        masks = generate_masks(mask_generator, image_array) 
        results[filename] = masks

        for i, mask in enumerate(masks):
            mask_image = Image.fromarray((mask['segmentation'].astype(np.uint8)) * 255)  
            mask_filename = os.path.join(mask_output_dir, f"mask_{i}_{filename.replace('.jpg', '.png')}")
            mask_image.save(mask_filename)
            print(f"Saved mask: {mask_filename}")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        show_anns(masks, ax) 
        ax.axis('off')
        ax.set_title(filename)

        output_path = os.path.join(output_dir, f"masked_{filename}")
        print(f"Saving to {output_path}")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)  

    return results


def main():
    image_dir = "./input_images" 
    output_dir = "./segmented_images" 
    model_cfg = "/configs/sam2.1/sam2.1_hiera_s.yaml"
    checkpoint = "./sam2/checkpoints/sam2.1_hiera_small_0.3.pth" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the SAM2 model
    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.2,
        mask_threshold=0.0,
        min_mask_region_area=0,
        output_mode="binary_mask",
        multimask_output=False
    )

    os.makedirs(output_dir, exist_ok=True)
    results = process_images(image_dir, mask_generator, output_dir)

    print("Inference complete! Masks have been generated and saved.")

if __name__ == "__main__":
    main()
