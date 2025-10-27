"""
GPU-optimized training script for microlensing CNN classifier
TimeDistributed architecture PRESERVED for real-time classification
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, TimeDistributed, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import argparse
import os
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    # Enable mixed precision for faster training
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
        # Enable memory growth to avoid OOM errors
        tf.config.experimental.set_memory_growth(gpu, True)
    
    return len(gpus)

def build_model(sequence_length=1500, num_channels=1, num_classes=2):
    """
    Build 1D CNN model with TimeDistributed layers
    TimeDistributed is ESSENTIAL for real-time classification
    where we assume data is coming in sequentially
    """
    model = Sequential([
        # First convolutional block
        Conv1D(filters=128, kernel_size=5, activation="relu", padding="same", 
               input_shape=(sequence_length, num_channels)),
        Dropout(0.3),
        
        # Second convolutional block
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        Dropout(0.3),
        
        # Third convolutional block
        Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
        Dropout(0.3),
        
        # TimeDistributed layers for sequential classification
        # This allows the model to make predictions at each time step
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(num_classes, activation="softmax", dtype='float32'))
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def preprocess_data(X_train, X_val, X_test):
    """Standardize features"""
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_train, n_time, n_feat = X_train.shape
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, n_time, n_feat)
    
    n_val = X_val.shape[0]
    X_val_2d = X_val.reshape(-1, n_feat)
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, n_time, n_feat)
    
    n_test = X_test.shape[0]
    X_test_2d = X_test.reshape(-1, n_feat)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_time, n_feat)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def create_callbacks(output_path, log_dir):
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    return callbacks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--output', required=True, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--experiment_name', type=str, default='baseline')
    args = parser.parse_args()
    
    # Setup GPUs
    print("=" * 80)
    print("GPU SETUP")
    print("=" * 80)
    num_gpus = setup_gpu()
    
    # Create multi-GPU strategy if available
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy with {num_gpus} GPUs')
    else:
        strategy = tf.distribute.get_strategy()
        print('Using default strategy (single GPU or CPU)')
    
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    X = data['X']
    y = data['y']
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Encode labels
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Split data: 70% train, 15% val, 15% test
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Preprocess
    print("=" * 80)
    print("PREPROCESSING")
    print("=" * 80)
    X_train, X_val, X_test, scaler = preprocess_data(X_train, X_val, X_test)
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    # Repeat labels for TimeDistributed (CRITICAL for real-time classification)
    print("Preparing labels for TimeDistributed architecture...")
    y_train_repeated = np.repeat(y_train_cat[:, np.newaxis, :], X_train.shape[1], axis=1)
    y_val_repeated = np.repeat(y_val_cat[:, np.newaxis, :], X_val.shape[1], axis=1)
    y_test_repeated = np.repeat(y_test_cat[:, np.newaxis, :], X_test.shape[1], axis=1)
    
    print(f"Training labels shape: {y_train_repeated.shape}")
    
    # Build model within strategy scope for multi-GPU
    print("=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    with strategy.scope():
        model = build_model(
            sequence_length=X_train.shape[1],
            num_channels=X_train.shape[2]
        )
    
    model.summary()
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * num_gpus if num_gpus > 1 else args.batch_size
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(args.output), f"logs_{args.experiment_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    callbacks = create_callbacks(args.output, log_dir)
    
    # Train
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}...")
    
    history = model.fit(
        X_train, y_train_repeated,
        validation_data=(X_val, y_val_repeated),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    test_loss, test_acc = model.evaluate(X_test, y_test_repeated, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save results
    results = {
        'experiment_name': args.experiment_name,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'epochs_trained': len(history.history['loss']),
        'batch_size': args.batch_size,
        'num_gpus': num_gpus,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    
    results_file = args.output.replace('.keras', '_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model saved to {args.output}")
    print(f"Results saved to {results_file}")
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
