import logging
import tensorflow as tf
import signal
import os
import matplotlib.pyplot as plt
import numpy as np
from datagen import train_generator, validation_generator, img_height, img_width
from plot import plot_training_history


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam





#Verify GPU availability
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    print("prova")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#Configure logging
logging.basicConfig(
    filename="Logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (5, 5), activation="relu", input_shape=(img_height, img_width, 3), padding="same"),
    Conv2D(32, (5, 5), activation="relu"),
    MaxPooling2D((2, 2), padding="same"),
    
    # Block 0
    Conv2D(64, (5, 5), activation="relu", padding="same"),
    Conv2D(64, (5, 5), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding="same"),
    
    # Block 1
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding="same"),
    
    # Block 2
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding="same"),
    
    # Block 3
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding="same"),
    
   
    # Flatten
    Flatten(),
    Dense(1024, activation="relu"),
    #Dropout(0.2),
    Dense(512, activation="relu"),
    Dense(8, activation="softmax")
])

if os.path.exists(r"models/best_model.keras"):
    print("Restarting from an existing model")
    model.load_weights(r"models/best_model.keras")
    print("Model loaded")

# Compile the model
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])

model.summary()




# Define a callback to log training information
class TrainingLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            "Epoch %d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f"
            % (
                epoch + 1,
                logs["loss"],
                logs["accuracy"],
                logs["val_loss"],
                logs["val_accuracy"],
            )
        )

# Create an instance of the training logger callback
training_logger_callback = TrainingLoggerCallback()



# Define the callback for model checkpointing
checkpoint_callback = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

# Define the callback for early stopping
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

# Train the model 
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[training_logger_callback, early_stopping_callback, checkpoint_callback],
    batch_size=64,
)

plot_training_history(history)