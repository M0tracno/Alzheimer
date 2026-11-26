import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt

def download_oasis_dataset():
    """
    Download and extract OASIS dataset
    Note: This is a simplified version. In practice, you need proper credentials and access.
    """
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    # You would typically download from OASIS website after proper authentication
    # For this example, we'll assume data is downloaded and placed in dataset/OASIS
    print("Please download the OASIS dataset from https://www.oasis-brains.org/ and place it in dataset/OASIS")

IMAGE_SIZE = (128, 128)
MODEL_PATH = 'models/resnet50_alzheimer.keras'


def entropy(counts):
    """Shannon entropy (bits) of a histogram of counts."""
    p = np.asarray(counts, dtype=np.float64).ravel()
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def calculate_mutual_information(x, y, bins=10):
    """Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y) between two arrays, in bits."""
    c_xy = np.histogram2d(np.ravel(x), np.ravel(y), bins)[0]
    return entropy(c_xy.sum(axis=0)) + entropy(c_xy.sum(axis=1)) - entropy(c_xy)

def preprocess_image(image):
    """Preprocess MRI scan"""
    # Normalize to 0-1 range (a blank image would otherwise divide by zero)
    image = (image - image.min()) / max(image.max() - image.min(), 1e-8)
    
    # Add channel dimension if needed (tf.image.resize needs one)
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)

    image = tf.image.resize(image, IMAGE_SIZE)
    
    return image

class MutualInformationLayer(layers.Layer):
    """Custom layer to incorporate mutual information"""
    def __init__(self, units=32, **kwargs):
        super(MutualInformationLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
    
    def call(self, inputs):
        # Calculate mutual information-based attention
        attention = tf.matmul(inputs, self.kernel)
        attention = tf.nn.softmax(attention, axis=-1)
        return tf.multiply(inputs, attention)

def create_model():
    """ResNet-50 (ImageNet, frozen) + dense head: the architecture served by app.py."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False

    return models.Sequential([
        layers.Input((*IMAGE_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(2048, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax'),  # classes in alphabetical folder order
    ])

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    """Train the Alzheimer's detection model"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Load and prepare the training dataset
    train_generator = train_datagen.flow_from_directory(
        'dataset/Combined Dataset/train',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # Load and prepare the validation dataset
    validation_generator = val_datagen.flow_from_directory(
        'dataset/Combined Dataset/train',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def evaluate_model(model):
    """Evaluate the trained model"""
    # Data generator for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and prepare the test dataset
    test_generator = test_datagen.flow_from_directory(
        'dataset/Combined Dataset/test',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = predictions.argmax(axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Calculate and print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train model
    model, history = train_model()
    
    # Evaluate model
    evaluate_model(model)
    
    print(f"\nModel saved successfully at {MODEL_PATH}")