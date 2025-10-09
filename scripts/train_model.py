import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import create_data_generators, build_model, get_class_names

# Configuration
DATA_DIR = '../data/garbage_dataset'
MODEL_SAVE_PATH = '../models/model.h5'
CLASS_NAMES_PATH = '../models/class_names.npy'
HISTORY_PLOT_PATH = '../outputs/training_history.png'
CONFUSION_MATRIX_PATH = '../outputs/confusion_matrix.png'
IMG_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 32

def train_model():
    """
    Complete training pipeline for garbage classification.
    """
    print("=" * 60)
    print("GARBAGE CLASSIFICATION MODEL - TRAINING PHASE")
    print("=" * 60)
    
    # Step 1: Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        print("Please download the dataset and extract it to the data/garbage_dataset folder")
        return
    
    print("\n[1/5] Loading data generators...")
    train_generator, val_datagen = create_data_generators(DATA_DIR, IMG_SIZE)
    
    # Get class names and count
    class_names = sorted(os.listdir(DATA_DIR))
    num_classes = len(class_names)
    print(f"     Found {num_classes} garbage categories:")
    for i, cls in enumerate(class_names, 1):
        print(f"     {i}. {cls.upper()}")
    
    # Save class names for later use in prediction
    np.save(CLASS_NAMES_PATH, np.array(class_names))
    print(f"     Class names saved to {CLASS_NAMES_PATH}")
    
    print("\n[2/5] Building transfer learning model...")
    model = build_model(num_classes, IMG_SIZE)
    print("     Model created successfully!")
    print("     Architecture:")
    print(f"     - Input Layer: {IMG_SIZE}×{IMG_SIZE}×3")
    print("     - Pre-trained MobileNetV2 layers (frozen)")
    print("     - Global Average Pooling: Reduces dimension")
    print("     - Dense(256) + Dropout(0.5)")
    print("     - Dense(128) + Dropout(0.5)")
    print(f"     - Output Layer: {num_classes} classes (softmax)")
    
    print("\n[3/5] Training the model...")
    print(f"     Epochs: {EPOCHS}")
    print(f"     Batch Size: {BATCH_SIZE}")
    print("     This may take 10-30 minutes depending on your hardware...")
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        verbose=1
    )
    
    print("\n[4/5] Saving trained model...")
    model.save(MODEL_SAVE_PATH)
    print(f"     Model saved to {MODEL_SAVE_PATH}")
    print(f"     Model size: {os.path.getsize(MODEL_SAVE_PATH) / 1024 / 1024:.2f} MB")
    
    print("\n[5/5] Generating visualizations...")
    plot_training_history(history, HISTORY_PLOT_PATH)
    print(f"     Training history plot saved to {HISTORY_PLOT_PATH}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Model saved: {MODEL_SAVE_PATH}")
    print(f"2. To make predictions, run: python predict.py --image <path_to_image>")
    print("=" * 60)

def plot_training_history(history, save_path):
    """
    Plots and saves training accuracy and loss.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], marker='o', label='Training Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], marker='o', color='red', label='Training Loss', linewidth=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"     Saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    train_model()