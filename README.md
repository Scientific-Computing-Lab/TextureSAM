
# TextureSAM
**TextureSAM** is a texture-aware variant of the Segment Anything Model (SAM), fine-tuned on the ADE20K dataset with texture augmentations. While SAM-2.1 relies on semantic cues, TextureSAM enhances segmentation in texture-dominant scenarios by introducing textures from the Describable Textures Dataset (DTD). Two versions are trained: one that preserves semantic structure and another that fully replaces objects with textures. Evaluated on RWTD (a natural texture-based segmentation dataset) and STMD (a synthetic texture-transition dataset), TextureSAM outperforms SAM-2.1 in capturing texture-based boundaries.

This code is forked and highly based on [**SAM2**](https://github.com/facebookresearch/sam2/tree/main/training) repository by Meta.

## Contents
```
1 Usage
2 Datasets
   2.1 Kaust256 - Real-World Textures Dataset (RWTD)
   2.2 ADE20K Dataset
   2.3 Synthetic Textured Masks Dataset (STMD)
3 Checkpoints
4 Usage Instructions
5 Training
6 Inference
7 Evaluation
8 🔥TextureSAM vs. SAM-2: Segmentation Performance
   - **STMD Results**
   - **RWTD Results**
   - **ADE20K Results**
9 Conclusions
```

## 1 Usage
Using this code, you can:
 - Fine-tune the SAM2 model on the ADE20K dataset and other custom texture datasets.
 - Run inference using the fine-tuned SAM2 model, called TextureSAM, to perform textural segmentation on the provided datasets.
 - Evaluate the model on the RWTD, STMD, and ADE20K datasets.
   
## 2 Datasets
The datasets used for the evaluation and inference of **TextureSAM** are available at the following link: [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link). 

The following datasets are included: 
### 2.1 Kaust256 - Real-World Textures Dataset (RWTD)
The **RWTD** dataset is a collection of real-world images specifically curated to evaluate segmentation models based on texture cues rather than object boundaries. Unlike conventional segmentation datasets that focus on semantic elements, RWTD emphasizes regions where segmentation is defined by texture changes. This dataset provides a realistic test case for models like TextureSAM, which are designed to mitigate the shape bias observed in SAM-2.
 - Kaust256 contains 256 images.
 - **Textured images** are also provided, generated using an ADE20K-based filtering process.
 - Labels are binary segmentation masks provided in PNG format.
```
    |-- Kaust256
        |-- images
        |   |-- 101.jpg
        |   |-- 102.jpg
        |   |-- 103.jpg
        |   |-- ...
        |-- labels
        |   |-- 101.png
        |   |-- 102.png
        |   |-- 103.png
        |   |-- ...
        |-- images_textured
            |-- 101.jpg
            |-- 102.jpg
            |-- 103.jpg
            |-- ...
```

### 2.2 ADE20K Dataset
The **ADE20K** dataset is widely used for scene parsing and semantic segmentation. It includes over 20,000 images with dense annotations spanning 150 semantic categories. To facilitate texture-aware segmentation, multiple versions of ADE20K have been generated:
1. **ADE20K_0.3** : ADE20K dataset modified with texture augmentation (η ≤ 0.3), provided in the SAM-compatible format. Each image is paired with a corresponding annotation JSON file, making it suitable for fine-tuning SAM to balance texture and semantic segmentation.
2. **ADE20K_real_images**: The original ADE20K dataset without modifications.
This directory contains real ADE20K images along with their corresponding ground truth (GT) instance masks. The dataset is organized into two main subdirectories:
 - ADE20K_unified_gt_instances: Stores unified ground truth instance masks.
 - images: Stores the original ADE20K images.
 - gt_instances: Contains individual instance segmentation masks for each image.
 - 
3. **ADE20K_textured_images**: Includes ADE20K images with all degrees of texture augmentation (η = 0 to 1) . These images are designed to decouple shape and texture by replacing object semantics with textures sampled from the **Describable Textures Dataset (DTD)**.

The **evaluation** of TextureSAM was conducted using the **ADE20K_real_images**.

```
   ├── ADE20K_0.3/
   │   ├── ADE_train_00000037_textured_degree0_0.jpg
   │   ├── ADE_train_00000037_textured_degree0_0.json
   │   ├── ADE_train_00000037_textured_degree0_1.jpg
   │   ├── ADE_train_00000037_textured_degree0_1.json
   │   ├── ADE_train_00000037_textured_degree0_2.jpg
   │   ├── ADE_train_00000037_textured_degree0_2.json
   │   ├── ADE_train_00000037_textured_degree0_3.jpg
   │   ├── ADE_train_00000037_textured_degree0_3.json
   │   ├── ADE_train_00000037.jpg
   │   ├── ADE_train_00000037.json
   │   └── ...

```

```
ADE20K_real_images/
    ├── ADE20K_unified_gt_instances/
    │   ├── unified_ADE_val_00000001.png
    │   ├── unified_ADE_val_00000002.png
    │   ├── unified_ADE_val_00000003.png
    │   └── ...
    ├── gt_instances/
    │   ├── instance_000_ADE_val_00000001.png
    │   ├── instance_001_ADE_val_00000001.png
    │   ├── instance_000_ADE_val_00000002.png
    │   ├── instance_001_ADE_val_00000002.png
    │   └── ...
    ├── images/
    │   ├── ADE_val_00000001.jpg
    │   ├── ADE_val_00000002.jpg
    │   ├── ADE_val_00000003.jpg
    │   └── ...
```

### 2.3 Synthetic Textured Masks Dataset (STMD)
The STMD dataset is a synthetic benchmark that evaluates segmentation models in a controlled texture-only environment. The **Test Datset** is available at the following link: [**Segmentation_dataset**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link).
Unlike RWTD, this dataset eliminates object boundaries and contains images composed purely of texture transitions. By isolating segmentation performance to texture cues alone, STMD provides a rigorous assessment of a model's ability to distinguish regions based solely on texture variations. The full STMD dataset is available on GitHub:  [**STMD**](https://github.com/mubashar1030/Segmentation_dataset) .
 - Segmentation dataset contains 10,000 masks.
 - Each mask has 5 images with random textures for each region, resulting in a total of 50,000 images.
 - Training Set: 42,000 images.
 - Validation Set: 3,000 images.
 - Test Set: 5,000 images.

The [**Segmentation_dataset**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link) stored in Google Drive is organized as follows:
 - images_test/ – Contains test images used for model evaluation.
 - images_textured/ – Includes images processed with the ADE20K filter to enhance texture-based segmentation challenges.
 - labels_test/ – Stores the segmentation masks corresponding to the test images, serving as ground truth for evaluation.
 - test.txt – Lists the file paths of all test images, ensuring consistency in dataset loading.
 - train.txt – Contains the file paths for training images.
 - val.txt – Includes the file paths for validation images, used to monitor performance during training.
 
 The **evaluation** of TextureSAM was conducted using the images_test/ and labels_test/ . The test.txt file was used to define the dataset split.
```
Segmentation_dataset/
│── images_test/
│   ├── 7.jpg
│   ├── 17.jpg
│   ├── 18.jpg
│   └── ...
│── labels_test/
│   ├── 7.png
│   ├── 17.png
│   ├── 18.png
│   └── ...
```

## 3 Checkpoints:
This repo provides three fine-tuned SAM-2.1 checkpoints, available in [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link):
 - **sam2.1_hiera_small_0.3.pt** – The best-performing **TextureSAM model** for texture segmentation, fine-tuned for 19 epochs with mild texture augmentation 
   (η ≤ 0.3), achieving the best balance between texture and semantic segmentation.
 - **sam2.1_hiera_small_1.pt** – Fine-tuned for 25 epochs with strong texture augmentation (η ≤ 1.0), optimized for texture-based segmentation.
 - **sam2.1_hiera_small.pt** – Pre-trained SAM-2.1 Hiera Small, used as the base model before fine-tuning.

```
Checkpoints/
│── sam2.1_hiera_small.pt
│── sam2.1_hiera_small_0.3.pt
│── sam2.1_hiera_small_1.pt
│── download_ckpts.sh 
```
To download the original SAM-2.1 model, use download_ckpts.sh.

## 4 Usage Instructions:
- **Set up the Environment for Running TextureSAM:**
   Use the conda configuration file created by TextureSAM environment setup, located at texturesam.yaml.
   Set up the environment by running the following command:
   ```python
   conda env create --name sam -f texturesam.yaml
   ```
- **Activate the sam environment:**
   ```python
   conda activate texturesam
   ```
## 5 Training 
**Requirements**: We assume training is conducted on A100 GPUs with 80 GB of memory
### Steps to Fine tune SAM-2.1 SMALL on ADE20K to Produce TextureSAM 
  - **Download the SAM-2.1 Small Checkpoint**
    Run the following commands to download the checkpoint:
    ```
    cd sam2/checkpoints && \
    ./download_ckpts.sh && \
    cd ..
    ```
 - **Download the dataset**
    - Get the **ADE20K_0.3** dataset from the  [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe- 
      CsPkwRv?usp=drive_link).
   - Set the paths for both ADE20K_0.3 dataset and SAM-2.1 Small checkpoint in sam2/sam2/configs/sam2.1_training/sam2.1_hiera_s_finetune.yaml.
    - This dataset is already formatted for SAM 2 training.
 - **Fine-tune the model**
   To fine-tune the **SAM-2.1 Small** model on the ADE20K dataset as was done in the paper using 1 GPUs, run the following command:
     ```
     python sam2/training/train.py \
      -c sam2/sam2/configs/sam2.1_training/sam2.1_hiera_s_finetune.yaml \
      --use-cluster 0 \
      --num-gpus 1
     ```
- You can also train SAM 2.1 Huge or Base by modifying the configuration file accordingly.
- **Training setup used in the paper**:
    In the paper, we trained the model using a single A100 GPU. 
- The training process follows the detailed steps in sam2/training, which is a fork of the original  [**SAM 2 training instructions**](https://github.com/facebookresearch/sam2/tree/main/training).

## 6 Inference
   To run **inference** using TextureSAM or SAM on a dataset of your choice, use the sam2/inference.py script. This script processes input images and 
   generates segmented output images along with their corresponding binary masks in a subfolder called masks_output.
 
 ### Steps to Run Inference
  - **Set Up Paths**
    Before running the script, update the following paths in **sam2/inference.py** to match your dataset and model configuration:
      - image_dir: Path to your input images.
      - output_dir: Path where segmented images and masks will be saved.
      - model_cfg: Configuration file for the model.
      - checkpoint: Path to the model checkpoint.
  - Run the **Inference Script:**    ```  python sam2/inference.py     ```

  - In our implementation, we **adjusted the parameters** of the automatic mask generator to optimize segmentation results.
     - Increased the number of sampling points to capture more detailed features.
     - Set a higher IoU threshold to ensure high-confidence predictions.
     - Adjusted stability and mask thresholds to control segmentation sensitivity.
     - Configured the model to produce single binary masks per region for cleaner outputs.
     - These changes enhance segmentation performance and were applied as described in the paper.

## 7 Evaluation     
To **evaluate** the the performance of TextureSAM against SAM, we provide evaluation scripts for different datasets and evaluation settings. These scripts compute **Mean Intersection over Union (mIoU)** and, for non-aggregated masks, also use the **Adjusted Rand Index (ARI)** to measure segmentation quality.

### Evaluation Scripts
1. ``` eval_agg_masks.py ```
  - Evaluates segmentation with **aggregated masks**  for the RWTD and STMD datasets, merging overlapping predicted masks before computing **mIoU**, and matches predicted masks to ground truth labels based on pixel overlap for accurate evaluation.
  - For **RWTD**, mIoU is averaged over both original and inverted labels due to binary label inconsistencies.
  - **Paths to set before running:**
      - **Ground truth labels directory**: label_dir = "./ground_truth_labels"
      - **Predicted masks** pred_masks_dir = "./masks_output" (folder with predicted segmentations)
      - **Output results file**: output_csv_path = "./evaluation_results.csv" (stores evaluation results)
      - **RWTD dataset flag**: rwtd_dataset = False (set to True for RWTD dataset evaluation)
2. ``` eval_agg_masks_ADE20K.py ```
  - Evaluates segmentation with **aggregated masks** for the **ADE20K** dataset.
  - Uses **instance unification** by merging predicted masks into a single segmentation mask before evaluation.
  - Matches predicted masks to ground truth labels based on pixel overlap to improve evaluation accuracy.
  - **Paths to set before running:**
       - **Unified ground truth labels:** unified_labels_dir = "./unified_labels" (Directory containing ADE20K unified ground truth masks, stored in [**TextureSAM Datasets and Checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link) under 
     `ADE20k_real_images/ADE20K_unified_gt_instances`.   
       - **Predicted masks directory:** pred_root_dir = "./pred_masks" (folder with predicted segmentation masks)
       -  **Output results file:** output_csv = "./evaluation_results.csv" (stores evaluation results)
  - The script processes all images in the dataset, computes **mIoU scores** for each sample, and saves results in evaluation_results.csv.
    
3. ``` eval_no_agg_masks.py ```
  - Evaluates segmentation **without aggregated masks**, comparing each predicted mask separately.
  - Computes both **Mean IoU (mIoU)** and **Adjusted Rand Index (ARI)** to assess segmentation quality.
  - Penalizes over-segmentation by evaluating individual predicted masks instead of merging overlapping ones.
  - **Paths to set before running:**
       - **Ground truth labels directory**: gt_folder = "./path_to_ground_truth_masks"
       - **Predicted masks directory:** pred_folder = "./masks_output" (Folder with predicted segmentation masks.)
       - **Output results file:** output_csv = "./evaluation_results.csv" (Stores evaluation results.)
  - The script processes all images, calculates mIoU and ARI per sample, and saves results in evaluation_results.csv.
    
4. ``` eval_no_agg_masks_ADE20K.py ```
  - Evaluates segmentation **without aggregated masks** for the **ADE20K dataset**.
  - Computes both **Mean IoU (mIoU)** and **Adjusted Rand Index (ARI)** to measure segmentation quality.
  - Assesses the impact of over-segmentation by evaluating individual predicted masks instead of merging overlapping ones.
  - **Paths to set before running:**
       - **Unified ground truth labels**: gt_folder = "./path_to_ground_truth_masks"(Directory containing ADE20K unified ground truth masks.)
       - **Predicted masks directory:** pred_folder = "./masks_output" (Folder with predicted segmentation masks.)
       - **Output results file:** output_csv = "./evaluation_results.csv" (Stores evaluation results.)
  - The script processes all images in the dataset, computes mIoU and ARI per ground truth mask, and saves results in evaluation_results.csv.

## 🔥🔥4 TextureSAM vs. SAM: Segmentation Performance Across Datasets🔥🔥
  **What is SAM-2 and SAM-2?**
   - **SAM-2**: The original **Segment Anything Model (SAM)** variant, primarily relying on shape cues for segmentation, making it less effective in texture-based regions.
   - **SAM-2***: A **modified version** of SAM-2, using TextureSAM’s inference parameters to reduce shape bias and align its evaluation settings for a direct comparison.

|   STMD Results    | mIoU  | ARI  | mIoU (Aggregated)   |
| -----------| ---- | ---- | ---- |
|SAM-2|0.07 |0.16 | 0.16
|SAM-2*|0.17 |0.16 | **0.78**
|TextureSAM η ≤ 0.3|0.33 |0.32 |0.71 
|TextureSAM η ≤ 1|**0.35** |**0.34**|0.7 

|   RWTD Results    | mIoU  | ARI  | mIoU (Aggregated)   |
| -----------| ---- | ---- | ---- |
|SAM-2|0.26 |0.36 | 0.44
|SAM-2*|0.14 |0.19 | 0.75
|TextureSAM η ≤ 0.3|**0.47** |**0.62** |0.75 
|TextureSAM η ≤ 1|0.42 |0.54|**0.76** 

|   ADE20k Results    | mIoU  | ARI  | mIoU (Aggregated)   |
| -----------| ---- | ---- | ---- |
|SAM-2|0.08 |0.08 | 0.46
|SAM-2*|0.1 |0.11 | **0.65**
|TextureSAM η ≤ 0.3|0.11 |0.11 |0.55 
|TextureSAM η ≤ 1|0.12 |0.11|0.40

## 5 Conclusions
- **TextureSAM reduces SAM-2's shape bias**, achieving superior segmentation in texture-based datasets (STMD & RWTD).
- Fine-tuning with texture augmentations shifts focus from shape to texture, balancing segmentation performance.
 






