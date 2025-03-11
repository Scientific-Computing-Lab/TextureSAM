
# TextureSAM
**TextureSAM** is a texture-aware variant of the Segment Anything Model (SAM), fine-tuned on the ADE20K dataset with texture augmentations. While SAM-2.1 relies on semantic cues, TextureSAM enhances segmentation in texture-dominant scenarios by introducing textures from the Describable Textures Dataset (DTD). Two versions are trained: one that preserves semantic structure and another that fully replaces objects with textures. Evaluated on RWTD (a natural texture-based segmentation dataset) and STMD (a synthetic texture-transition dataset), TextureSAM outperforms SAM-2.1 in capturing texture-based boundaries.

This code is forked and highly based on [**SAM2**](https://github.com/facebookresearch/sam2/tree/main/training) repository by Meta.

## 1 Datasets
The datasets used for the evaluation and inference of **TextureSAM** are available at the following link: [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link). 

The following datasets are included: 
### 1.1 Kaust256 - Real-World Textures Dataset (RWTD)
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

### 1.2 ADE20K Dataset
The **ADE20K** dataset is widely used for scene parsing and semantic segmentation. It includes over 20,000 images with dense annotations spanning 150 semantic categories. To facilitate texture-aware segmentation, multiple versions of ADE20K have been generated:
1. **ADE20K_0.3** : ADE20K dataset modified with texture augmentation (η ≤ 0.3), provided in the SAM-compatible format. Each image is paired with a corresponding annotation JSON file, making it suitable for fine-tuning SAM to balance texture and semantic segmentation.
2. **ADE20K_real_images**: The original ADE20K dataset without modifications.
This directory contains real ADE20K images along with their corresponding ground truth (GT) instance masks. The dataset is organized into two main subdirectories:
 - ADE20K_unified_gt_instances: Stores unified ground truth instance masks.
 - images: Stores the original ADE20K images.
 - gt_instances: Contains individual instance segmentation masks for each image.

ADE20K_unified_gt_instances: Stores unified ground truth instance masks.
gt_instances: Contains individual instance segmentation masks for each image.
images: Stores the original ADE20K images.
4. **ADE20K_textured_images**: Includes ADE20K images with all degrees of texture augmentation (η = 0 to 1) . These images are designed to decouple shape and texture by replacing object semantics with textures sampled from the **Describable Textures Dataset (DTD)**.

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

### 1.3 Synthetic Textured Masks Dataset (STMD)
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

### Using this code, you can:
- Fine-tune TextureSAM on custom texture datasets, including ADE20K.
- Evaluate the model on RWTD, STMD, and ADE20K datasets.
- Run inference using fine-tuned models to segment texture-based regions in the provided datasets.

## Usage Instructions:
1. **Set up the Environment for Running TextureSAM:**
   Use the conda configuration file created by TextureSAM environment setup, located at TextureSAM/texturesam.yaml.
   Set up the environment by running the following command:
   ```python
   conda env create --name sam -f TextureSAM/texturesam.yaml
   ```
3. **Activate the sam environment:**
   ```python
   conda activate texturesam
   ```
4. Load the TextureSAM checkpoint **sam2.1_hiera_small_0.3.pt** and the relevant dataset from the following link: [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link) .
   
5. Place the sam2.1_hiera_small_0.3.pt file in the directory TextureSAM/sam2/checkpoints/.

6. To **train** SAM 2 as described in the paper on ADE20K to produce TextureSAM, load the **ADE20K_0.3** dataset from the  [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link). This dataset is in the correct format for SAM 2 training. Follow the full training instructions provided at TextureSAM/sam2/training which are a fork of the original [**SAM 2 training instructions**](https://github.com/facebookresearch/sam2/tree/main/training).
   
7. To run **inference** on TextureSAM, use the TextureSAM/sam2/inference.py script.
   This will process the input images and produce segmented output images along with corresponding binary masks.

8. Run **Evaluations** on the Provided Datasets:
   The following evaluation scripts can be used to assess the model's performance on the RWTD, STMD, and ADE20K datasets, both with and without aggregated masks:
   - eval_agg_masks.py
   - eval_agg_masks_ADE20K.py
   - eval_no_agg_masks.py
   - eval_no_agg_masks_ADE20K.py

- Make sure to set the **correct paths** to the input images, ground truth masks, and any other required resources in each script.





