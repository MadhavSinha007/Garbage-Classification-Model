import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_data_generators(train_dir, img_size=224):
    """
    Creates data generators for training and validation.
    Data augmentation helps the model generalize better.
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values to 0-1
        rotation_range=20,                 # Randomly rotate images
        width_shift_range=0.2,             # Randomly shift width
        height_shift_range=0.2,            # Randomly shift height
        zoom_range=0.2,                    # Randomly zoom
        horizontal_flip=True,              # Randomly flip horizontally
        fill_mode='nearest'                # Fill new pixels
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training images from subdirectories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical'  # Multi-class classification
    )
    
    return train_generator, val_datagen

def build_model(num_classes, img_size=224):
    """
    Builds a transfer learning model using MobileNetV2.
    
    MobileNetV2: A lightweight model good for mobile/edge devices.
    Pre-trained on ImageNet (1.4M images, 1000 categories).
    """
    # Load pre-trained MobileNetV2 without top classification layer
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze pre-trained weights (we don't want to retrain them)
    base_model.trainable = False
    
    # Add custom layers for garbage classification
    inputs = base_model.input
    x = GlobalAveragePooling2D()(base_model.output)  # Reduce dimensionality
    x = Dense(256, activation='relu')(x)              # Learn garbage features
    x = Dropout(0.5)(x)                               # Prevent overfitting
    x = Dense(128, activation='relu')(x)              # More feature learning
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Output probabilities
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # For multi-class classification
        metrics=['accuracy']
    )
    
    return model

def get_class_names(train_dir):
    """
    Extracts class names from subdirectory names.
    """
    class_names = sorted(os.listdir(train_dir))
    return class_names