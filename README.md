# 🗑️ Garbage Classification using Deep Learning

A deep learning-based image classification system that identifies types of garbage (e.g., plastic, glass, cardboard, metal, paper) using **MobileNetV2 transfer learning**.

---

## 🚀 Features

* 📸 Classifies garbage images into multiple categories
* ⚡ Uses lightweight MobileNetV2 (fast & efficient)
* 🧠 Transfer learning (pre-trained on ImageNet)
* 📊 Training visualization (accuracy & loss plots)
* 🔍 Simple CLI-based prediction system

---

## 📂 Project Structure

```
|── Data                      #Data set
├── models
│   ├── class_names.npy       # Saved class labels
│   └── model.h5              # Trained model
├── outputs
│   └── training_history.png  # Training graphs
├── scripts
│   ├── predict.py            # Prediction script
│   ├── train_model.py        # Training pipeline
│   └── utils.py              # Helper functions
├── test_images
│   ├── cardboard_sheets.jpeg
│   └── Clear-Glass.jpg
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

> ⚠️ Skip this if you already have `model.h5`

```bash
cd scripts
python train_model.py
```

### Output:

* Trained model → `models/model.h5`
* Class labels → `models/class_names.npy`
* Training plot → `outputs/training_history.png`

---

## 🔍 Running Predictions

### Step 1: Navigate to scripts folder

```bash
cd scripts
```

### Step 2: Run prediction

```bash
python predict.py --image ../test_images/cardboard_sheets.jpeg
```

### Example:

```bash
python predict.py --image ../test_images/Clear-Glass.jpg
```

---

## 📊 Sample Output

```
PREDICTION RESULTS

Image: ../test_images/cardboard_sheets.jpeg
Predicted Garbage Type: CARDBOARD
Confidence: 95.21%

All predictions:
  CARDBOARD: 95.21%
  GLASS: 2.10%
  PLASTIC: 1.34%
  METAL: 0.80%
  PAPER: 0.55%
```

* Displays image with predicted label

---

## ⚠️ Common Issues

### ❌ Model not found

Ensure path exists:

```
models/model.h5
```

### ❌ Image not found

Check path:

```
--image ../test_images/your_image.jpg
```

### ❌ Wrong working directory

Always run from:

```
scripts/
```
---

## 🧪 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* PIL

---

## 👨‍💻 Author

Madhav

---

## 📜 License

This project is open-source and available under the MIT License.
