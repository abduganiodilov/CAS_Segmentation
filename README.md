# CAS_Segmentation
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/abduganiodilov/CAS_Segmentation)

This repository provides a complete, command-line-driven pipeline for 2D semantic segmentation of 3D medical images. It handles the entire workflow from preprocessing NIfTI files to training, and evaluating a U-Net based segmentation model using PyTorch.

The pipeline is designed to be resumable and efficient, making it suitable for processing large datasets.

## Features

-   **Data Extraction**: Unzips dataset archives.
-   **3D to 2D Conversion**: Converts 3D NIfTI volumes (`.nii.gz`) into 2D PNG slices, automatically pairing images and masks.
-   **Resumable Slicing**: Tracks processed volumes to allow for chunked, interruptible preprocessing of large datasets.
-   **Automated Dataset Curation**: Generates a `dataset.csv` file mapping image slices to their corresponding masks.
-   **Model Training**:
    -   Trains a U-Net model with a ResNet-34 encoder using `segmentation-models-pytorch`.
    -   Includes data augmentations (flips, rotations, affine transforms) via `albumentations`.
    -   Features resumable training from the last checkpoint.
    -   Implements early stopping and a `ReduceLROnPlateau` learning rate scheduler to prevent overfitting and optimize training time.
-   **Model Evaluation**:
    -   Evaluates the best-performing model on a held-out test set.
    -   Calculates and saves Intersection over Union (IoU) and Dice scores.
    -   Generates a visual grid of predictions for qualitative assessment.
-   **Performance**: Optimized for speed with PyTorch's Automatic Mixed Precision (AMP) and efficient data loading.

## Requirements

To run this pipeline, you will need Python 3.8+ and the following packages. You can install them using pip:

```bash
pip install torch torchvision
pip install pandas numpy tqdm opencv-python
pip install nibabel scikit-learn segmentation-models-pytorch albumentations matplotlib
```

## Workflow and Usage

The pipeline is controlled via `imagecas_pipeline.py` with several commands. The following steps demonstrate the end-to-end workflow.

### 1. Extract Data

If your dataset is in a ZIP archive, extract it first.

```bash
python imagecas_pipeline.py extract --zip /path/to/your/dataset.zip --out ./data/raw_volumes
```
- `--zip`: Path to the input ZIP file.
- `--out`: Directory where the contents will be extracted.

### 2. Slice 3D Volumes to 2D Images

Convert the 3D `.nii.gz` volumes into 2D `.png` slices. The script automatically finds image-label pairs based on common naming conventions. This process is resumable; if interrupted, it will pick up where it left off.

```bash
python imagecas_pipeline.py slice --in ./data/raw_volumes --out ./data/slices --chunk-size 50
```
- `--in`: The directory containing the extracted 3D NIfTI volumes.
- `--out`: The target directory to save the 2D `images/` and `masks/` subdirectories.
- `--chunk-size`: (Optional) The number of volumes to process in one run. Defaults to 50.

### 3. Create Dataset CSV

Generate a `dataset.csv` file that lists the corresponding image and mask file paths. This file is used by the training and evaluation scripts.

```bash
python imagecas_pipeline.py make-csv --slices ./data/slices
```
- `--slices`: The directory containing the `images/` and `masks/` folders from the previous step.

### 4. Train the Segmentation Model

Train the U-Net model on the 2D slices. The script automatically splits the data, applies augmentations, and saves the best model.

```bash
python imagecas_pipeline.py train \
    --slices ./data/slices \
    --run-dir ./runs/experiment_01 \
    --epochs 30 \
    --batch-size 8 \
    --img-size 512 \
    --workers 4
```
- `--slices`: Path to the directory with `dataset.csv`.
- `--run-dir`: Directory to save model checkpoints (`last.pt`, `best.pt`).
- `--epochs`: Maximum number of training epochs.
- `--batch-size`: Number of images per batch.
- `--img-size`: Image size to which slices are resized.
- `--workers`: Number of data loader workers.

### 5. Evaluate the Model

Evaluate the best model from the training run on the test set. This will compute and print the final metrics and save them to `metrics.txt`.

```bash
python imagecas_pipeline.py eval \
    --slices ./data/slices \
    --run-dir ./runs/experiment_01 \
    --batch-size 8 \
    --img-size 512 \
    --preview-grid
```
- `--slices`: Path to the directory with `dataset.csv`.
- `--run-dir`: Directory containing the `best.pt` model checkpoint.
- `--preview-grid`: (Optional) Add this flag to generate and save a PNG image (`val_preds_grid.png`) showing a sample of images, ground truth masks, and model predictions.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
