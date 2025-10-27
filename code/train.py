"""
TensorFlow training script for microlensing CNN classifier
GPU-optimized for AMD MI300 with TimeDistributed architecture
Compatible with the rest of the thesis-microlens codebase
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, 
                                      TimeDistributed, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
import json
import joblib
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def setup_gpu():
    """Configure GPU settings for AMD MI300"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✓ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Enable mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✓ Mixed precision enabled (FP16)")
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠ WARNING: No GPUs detected! Training will be slow on CPU.")
    
    return len(gpus)

def build_model(sequence_length=1500, num_channels=1, num_classes=2):
    """
    Build TimeDistributed 1D CNN for real-time classification
    
    Architecture:
    - 3x Conv1D blocks with BatchNorm and Dropout
    - TimeDistributed Dense layers for per-timestep classification
    - Preserves temporal structure for early detection analysis
    """
    model = Sequential([
        # First convolutional block
        Conv1D(128, kernel_size=5, activation='relu', padding='same',
               input_shape=(sequence_length, num_channels)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second convolutional block
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third convolutional block
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # TimeDistributed classification head
        # This applies Dense layer at EACH timestep for real-time classification
        TimeDistributed(Flatten()),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dropout(0.3)),
        TimeDistributed(Dense(num_classes, activation='softmax'))
    ])
    
    # Compile with categorical crossentropy for classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def preprocess_data(X_train, X_val, X_test):
    """
    Standardize features using training set statistics
    """
    print("\nPreprocessing data...")
    
    # Flatten for scaling
    n_train, n_time, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    # Transform all sets
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_train, n_time, n_feat)
    
    n_val = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_feat)).reshape(n_val, n_time, n_feat)
    
    n_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, n_time, n_feat)
    
    print("✓ Data standardized")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def main():
    parser = argparse.ArgumentParser(description='Train microlensing classifier')
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--output', required=True, help='Path to save model (.keras)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--experiment_name', type=str, default='baseline', 
                       help='Experiment name for tracking')
    args = parser.parse_args()
    
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_base = os.path.dirname(os.path.dirname(args.output))
    experiment_dir = os.path.join(results_base, 'results', f'{args.experiment_name}_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    print("=" * 80)
    print("MICROLENSING BINARY CLASSIFICATION - TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Results directory: {experiment_dir}")
    print("=" * 80)
    
    # Setup GPU
    print("\nGPU SETUP")
    print("-" * 80)
    num_gpus = setup_gpu()
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Loading from: {args.data}")
    
    data = np.load(args.data, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    
    # Encode labels
    print("\nEncoding labels...")
    label_map = {'PSPL': 0, 'Binary': 1}
    y_encoded = np.array([label_map[label] for label in y])
    
    # Split data: 70% train, 15% val, 15% test
    print("\nSplitting data (70/15/15)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]:,} samples")
    print(f"Val:   {X_val.shape[0]:,} samples")
    print(f"Test:  {X_test.shape[0]:,} samples")
    
    # Preprocess
    X_train, X_val, X_test, scaler = preprocess_data(X_train, X_val, X_test)
    
    # Convert labels to categorical and repeat for TimeDistributed
    print("\nPreparing labels for TimeDistributed architecture...")
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)
    
    # Repeat labels for each timestep (TimeDistributed expects this)
    y_train_repeated = np.repeat(y_train_cat[:, np.newaxis, :], X_train.shape[1], axis=1)
    y_val_repeated = np.repeat(y_val_cat[:, np.newaxis, :], X_val.shape[1], axis=1)
    y_test_repeated = np.repeat(y_test_cat[:, np.newaxis, :], X_test.shape[1], axis=1)
    
    print(f"Train labels shape: {y_train_repeated.shape}")
    
    # Build model
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)
    
    model = build_model(
        sequence_length=X_train.shape[1],
        num_channels=X_train.shape[2],
        num_classes=2
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Callbacks
    print("\n" + "=" * 80)
    print("CONFIGURING CALLBACKS")
    print("=" * 80)
    
    model_path = os.path.join(experiment_dir, 'best_model.keras')
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("✓ Callbacks configured")
    
    # Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using {num_gpus} GPU(s)")
    print("-" * 80)
    
    history = model.fit(
        X_train, y_train_repeated,
        validation_data=(X_val, y_val_repeated),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    
    test_results = model.evaluate(X_test, y_test_repeated, verbose=0)
    
    print("\nTest Results:")
    print(f"  Loss:      {test_results[0]:.4f}")
    print(f"  Accuracy:  {test_results[1]:.4f}")
    print(f"  Precision: {test_results[2]:.4f}")
    print(f"  Recall:    {test_results[3]:.4f}")
    print(f"  AUC:       {test_results[4]:.4f}")
    
    # Save artifacts
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)
    
    # Save scaler
    scaler_path = os.path.join(experiment_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save configuration
    config = {
        'experiment_name': args.experiment_name,
        'timestamp': timestamp,
        'data_file': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_gpus': num_gpus,
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'sequence_length': int(X_train.shape[1]),
        'num_channels': int(X_train.shape[2]),
    }
    
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}")
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"✓ History saved: {history_path}")
    
    # Save results summary
    results = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'test_auc': float(test_results[4]),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
    }
    
    results_path = os.path.join(experiment_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {results_path}")
    
    # Copy model to specified output location
    if args.output != model_path:
        import shutil
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.copy(model_path, args.output)
        print(f"✓ Model copied to: {args.output}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Best model: {model_path}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test AUC: {test_results[4]:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
