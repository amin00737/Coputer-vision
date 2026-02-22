# A Hybrid Deep Learning-Based Approach in Agricultural Sustainability

## Apple Orchard Study

This repository contains the implementation of the hybrid detection–classification pipeline proposed in:

## Publications

- [My Paper on Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9XVrMOoAAAAJ&citation_for_view=9XVrMOoAAAAJ:u5HHmVD_uO8C)

---

## 📌 Overview

This project introduces a **hybrid deep learning framework** for:

1. **Apple Detection** using YOLO architectures
2. **Apple Ripeness Classification** using CNN and CNN-ML hybrid approaches

The system was developed for **smart orchard monitoring**, using RGB sensors installed in an apple orchard in Gvarv, Norway.

The final hybrid model:

* **YOLOv7 (Sub-dataset 3)** → 82% F1-score (Detection)
* **Custom CNN classifier** → 98.3% accuracy (Classification)
* 96.2% accuracy on external FruitNet dataset (Generality test)

---

## 🏗️ System Architecture

The pipeline consists of two modular components:

```
Raw Orchard Images
        ↓
YOLO Detector (v5 / v7 / v8)
        ↓
Detected & Cropped Apples
        ↓
Classifier (Fine-tuned CNN / CNN+ML / Custom CNN)
        ↓
Ripeness Class (0 / 1 / 2)
```

### Why Modular?

* Improves flexibility
* Enables independent optimization
* Reduces computational load
* Facilitates deployment in real-time agricultural systems

---

# 🌳 Dataset Description

## Detection Dataset (Dataset 1)

Captured using:

* Raspberry Pi HQ Camera
* Arducam 6mm lens
* LED illumination
* Night-time image acquisition
* Resolution: 3040 × 4056

Three sub-datasets were designed:

| Sub-dataset | Labeling Strategy                        |
| ----------- | ---------------------------------------- |
| Sub-1       | All visible apples labeled               |
| Sub-2       | Most visible apples labeled              |
| Sub-3       | Only apples with >50% visibility labeled |

Sub-dataset 3 (smallest dataset) produced the best detection results.

---

## Classification Dataset (Dataset 2)

* 500 detected apples → Augmented to ~3100 images
* 3 ripeness classes:

| Class | Days Until Harvest     |
| ----- | ---------------------- |
| 0     | 75–90 days (Unripe)    |
| 1     | 15–75 days (Semi-ripe) |
| 2     | <15 days (Ripe)        |

15% of test data sourced from FruitNet for generalization.

---

# 🔍 Detection Stage

Three YOLO versions were evaluated:

* YOLOv5
* YOLOv7
* YOLOv8

### Training Strategy

Total training cycles:

```
4 epoch configs × 3 batch sizes × 3 YOLO versions × 3 sub-datasets = 108 runs
```

### Best Results

| Model  | Sub-dataset | F1-score |
| ------ | ----------- | -------- |
| YOLOv7 | Sub-3       | **82%**  |

Additional notes:

* Weight size: 74.8 MB
* Detection time: 0.12 s per 3040×4056 image
* Robust to background apples and variant differences

---

# 🍎 Classification Stage

Three classification strategies were compared:

---

## 1️⃣ Fine-Tuned CNN Models

* VGG16
* VGG19
* ResNet50
* ResNet50V2
* ResNet152
* MobileNetV2
* Xception

Best fine-tuned model:

| Model | Accuracy |
| ----- | -------- |
| VGG16 | 0.924    |

---

## 2️⃣ CNN Feature Extraction + ML Classifier

Feature extractors:

* VGG16
* ResNet50

Classifiers:

* XGBoost
* Random Forest
* Support Vector Machine

Best hybrid combination:

| Model           | Accuracy |
| --------------- | -------- |
| VGG16 + XGBoost | 0.923    |

---

## 3️⃣ Custom CNN (Proposed Model)

Architecture:

* Conv blocks: 32 → 64 → 128 → 256 filters
* ELU activation
* Batch Normalization
* Adamax optimizer
* Categorical Crossentropy loss
* Image size: 124×124
* 30 epochs
* Early stopping

### Final Results

| Dataset   | Accuracy  |
| --------- | --------- |
| Dataset 2 | **0.983** |
| FruitNet  | **0.962** |

The custom CNN outperformed all baseline models.

---

# 🚀 Installation

```bash
git clone https://github.com/amin00737/Computer-vision.git
cd Computer-vision
pip install -r requirements.txt
```

If requirements file is missing:

```bash
pip install tensorflow keras torch opencv-python numpy pandas matplotlib scikit-learn xgboost
```

---

# ▶️ Running the Pipeline

## Detection

Train YOLO model:

```bash
python train_yolo.py --dataset sub_dataset_3 --model yolov7
```

Inference:

```bash
python detect.py --weights best.pt --source test_image.jpg
```

---

## Classification

Train classifier:

```bash
python train_classifier.py --model custom_cnn
```

Evaluate:

```bash
python evaluate.py --dataset dataset2
```

---

# 📊 Key Contributions

* Novel **sub-dataset strategy** for improving YOLO performance
* Lightweight detector suitable for real-time orchard deployment
* Custom CNN achieving 98.3% classification accuracy
* Modular hybrid architecture for smart farming systems

---

# 📖 Citation

If you use this work, please cite:

```bibtex
@article{mavaddat2024hybrid,
  title={A Hybrid Deep Learning-Based Approach in Agricultural Sustainability: Apple Orchard Study},
  author={Mavaddat, Amin and Andrade, Fabio A. A. and Hjelmervik, Karl Thomas and Johannessen, Erik Andrew and Hovden, Christian},
  year={2024}
}
```

---

# 📜 License

This project is released under the MIT License.

