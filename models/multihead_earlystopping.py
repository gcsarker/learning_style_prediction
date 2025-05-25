import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import Callback

class MultiHeadEarlyStopping(Callback):
    def __init__(self, monitor_heads=['classification_head', 'regression_head'], patience=5):
        super().__init__()
        self.monitor_heads = monitor_heads
        self.patience = patience
        self.best_losses = {head: np.inf for head in monitor_heads}
        self.waits = {head: 0 for head in monitor_heads}
        self.best_weights = None  # To store the best model weights
        self.stop_training_flags = {head: False for head in monitor_heads}

    def on_train_begin(self, logs=None):
        # Save initial weights at the beginning of training
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for head in self.monitor_heads:
            val_loss = logs.get(f'val_{head}_loss')
            if val_loss is None:
                continue  # Skip if metric not found

            # Check if validation loss improved
            if val_loss < self.best_losses[head]:
                self.best_losses[head] = val_loss
                self.waits[head] = 0
                self.best_weights = self.model.get_weights()  # Save the best weights
            else:
                self.waits[head] += 1

            # Stop training for that head if patience exceeded
            if self.waits[head] >= self.patience:
                self.stop_training_flags[head] = True
                # print(f"Early stopping triggered for {head} at epoch {epoch + 1}")

        # Restore best weights if any head stopped
        if any(self.stop_training_flags.values()):
            # print("Restoring best weights...")
            self.model.set_weights(self.best_weights)

        # Stop full model training only if both heads should stop
        if all(self.stop_training_flags.values()):
            self.model.stop_training = True
