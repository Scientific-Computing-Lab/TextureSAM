
# TextureSAM
**TextureSAM** is a texture-aware variant of the Segment Anything Model (SAM), fine-tuned on the ADE20K dataset with texture augmentations. While SAM-2.1 relies on semantic cues, TextureSAM enhances segmentation in texture-dominant scenarios by introducing textures from the Describable Textures Dataset (DTD). Two versions are trained: one that preserves semantic structure and another that fully replaces objects with textures. Evaluated on RWTD (a natural texture-based segmentation dataset) and STMD (a synthetic texture-transition dataset), TextureSAM outperforms SAM-2.1 in capturing texture-based boundaries.

## Available Datasets:
The datasets used for the evaluation and inference of **TextureSAM** are available at the following link: [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link). 

The following datasets are included: 
 1.  Kaust256 – The Real World Textures Dataset **(RWTD)**, used for real-world texture-based segmentation evaluation.
 2.  ADE20K Dataset:
       - ADE20K_0.3.zip – ADE20K dataset modified with texture augmentation (η ≤ 0.3), suitable for fine-tuning SAM.
       - ADE20K_real_images – Contains original ADE20K images.
       - ADE20K_textured_images – Includes ADE20K images with all degrees of texture augmentation (η = 0 to 1).
 3.  Segmentation_dataset – The **STMD** synthetic test dataset, designed for evaluating texture-based segmentation.
     The full **STMD** dataset can be found on GitHub:  [**STMD**](https://github.com/mubashar1030/Segmentation_dataset).

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
4. Load the TextureSAM checkpoint **sam2.1_hiera_small_0.3.pt** from the following link: [**TextureSAM Datasets and checkpoints**](https://drive.google.com/drive/folders/1pUJLa898WYEcb4Y_sOaXsSVe-CsPkwRv?usp=drive_link).
   
5. Place the sam2.1_hiera_small_0.3.pt file in the directory TextureSAM/sam2/checkpoints/.

6. 




