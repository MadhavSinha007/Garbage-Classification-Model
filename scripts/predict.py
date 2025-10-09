import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = '../models/model.h5'
CLASS_NAMES_PATH = '../models/class_names.npy'
IMG_SIZE = 224

def load_image(image_path):
    """
    Loads and preprocesses an image for prediction.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array, img
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        return None, None

def predict_garbage(image_path):
    """
    Predicts garbage type from an image.
    """
    print("=" * 60)
    print("GARBAGE CLASSIFICATION - PREDICTION")
    print("=" * 60)
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first using: python train_model.py")
        return
    
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"ERROR: Class names not found at {CLASS_NAMES_PATH}")
        return
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print("\n[1/3] Loading model and data...")
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True)
    print(f"     Model loaded successfully!")
    print(f"     Classes: {', '.join(class_names)}")
    
    print("\n[2/3] Loading and preprocessing image...")
    img_array, original_img = load_image(image_path)
    if img_array is None:
        return
    print(f"     Image loaded and resized to {IMG_SIZE}×{IMG_SIZE}")
    
    print("\n[3/3] Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nImage: {image_path}")
    print(f"Predicted Garbage Type: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll predictions:")
    for i, cls in enumerate(class_names):
        print(f"  {cls.upper()}: {predictions[0][i]*100:.2f}%")
    print("=" * 60)
    
    # Visualize
    visualize_prediction(original_img, predicted_class, confidence)

def visualize_prediction(img, predicted_class, confidence):
    """
    Displays the image with prediction result.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class.upper()}\nConfidence: {confidence:.2f}%", 
              fontsize=14, fontweight='bold', color='green')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict garbage type from an image")
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    
    predict_garbage(args.image)