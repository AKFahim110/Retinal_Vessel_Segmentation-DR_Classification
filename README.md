
# 🔬 Retinal Vessel Segmentation and Diabetic Retinopathy Classification using Vessel Graph-Based U-Net (VGA-Net)

This project presents a comprehensive deep learning pipeline for **retinal vessel segmentation** and **diabetic retinopathy classification** based on a **modified VGA-Net** architecture. It combines **pixel-level feature extraction**, graph-level structure preservation, and **disease classification** using graph-based attention mechanisms. The thesis was completely done using Keggle notebook.


## 📁 Project Structure

```
RetinalVesselDR/
├── data/                       # DRIVE dataset structure and loading
│   ├── training/
│   └── test/
├── preprocessing/             # Image preprocessing and augmentation
├── models/                    # Define Models for the task, such as DRIU and VGA-Net architecture and supporting modules (HDC, GCN, AB-FFM)
├── Pixel level segmentation/  # Define DRIU VGG 16 model for pixel level segmentation task 
├── VGA-Net model/	       # Define VGA-Net supportive modules and model (eg, HDC, Down Sampler, AB-FFM)
├── Training/                  # Train the model using necessary training functions
├── evaluation/                # Evaluation metrics (AUC, confusion matrix, etc.)
├── classification/            # Data preparation and preprocessing for DR classification
├── ResNet50/		       # Download pre-trained ResNet50 model to use transfer learning for DR classification
├── evaluation/                # Evaluation metrics (AUC, confusion matrix, etc.)
├── visualization/             # CAM and prediction outputs
├── utils.py                   # Helper functions
└── README.md                  # Project documentation (this file)
```

---

## 🧪 Tasks and Pipelines

| Task Type                                | Notebook/Script             | Dataset Path             | Description                                               |
| ---------------------------------------- | --------------------------- | ------------------------ | --------------------------------------------------------- |
| **Vessel Segmentation**                  | `main_segmentation.ipynb`   | `data/DRIVE/`            | Segment retinal blood vessels using VGA-Net               |
| **DR Classification**                    | `main_classification.ipynb` | `data/DRIVEseg_&output/` | Classify diabetic retinopathy stages using TL             |
| **Evaluation & Visualization**           | Embedded in notebooks       | -                        | images with utilized matrics; display results             |

---

## ⚙️ Features & Modules

- ✅ **Modified DRIU** architecture:
- ✅ **Modified VGA-Net** architecture:
  - **HDC Module** – Hierarchical Dilated Convolution
  - **Graph Construction** – Vessel graph modeling
  - **Node Downsampler** – downsample module
  - **GCN Module** – Graph Convolutional Network
  - **AB-FFM** – Attentional Bidirectional ConvLSTM Feature Fusion Module


- 📈 **Evaluation Metrics**:
  - Models were evaluated(on both validation and test sets) using:

	✅ Accuracy

	✅ SE

	✅ SP

	✅ DICE

	✅ MCC

	✅ Confusion Matrices


- 📊 **Visualization**:
  - Vessel segmentation overlays
  - DR classification confidence maps

---


## 🚀 Quick Start

### ✅ Dataset Setup

Make sure the DRIVE dataset is placed correctly in:
```
data/DRIVE/
├── training/
└── test/
```


📁 Dataset:

Datasets used in this project are available here: 🔗 https://www.kaggle.com/datasets/akfahim110/akfahim-thesis-dataset


### ✅ Run on Google Colab or Kaggle

Clone or upload the notebook and run the cells in order. Please note that Keggle has been employed for this work

```bash
# For segmentation:
Run: main_file.ipynb

# For classification:
Run: main_file.ipynb
```
-- Note: Segmentation and Classification are in the same notebook



### ✅ Sample Execution Flow

1. Import libraries and set dataset path
2. Preprocess fundus images
3. Train VGA-Net for vessel segmentation
4. Extract vessel features and perform classification using segmented output
5. Evaluate and visualize results for both task

---


## 📦 Requirements

- Python 3.10+
- OpenCV
- NumPy
- scikit-image
- Matplotlib
- scikit-learn
- PyTorch / TensorFlow (depending on backend)

Use a notebook environment (Google Colab or Kaggle) with GPU acceleration enabled.

---

## 📜 Citation

If you use this code or build on it, please cite the corresponding thesis:

> Fahim, Adalat Khan. *Retinal Vessel Segmentation and Diabetic Retinopathy Classification Using Vessel Graph Based U-Net Approach.* Sichuan University,2025.
