import time
import numpy as np
import pandas as pd
import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Calibration settings
# --------------------------------------------------

DATA_PATH = (
    "/Users/emreyurgaagent/Desktop/substack/"
    "neural parallel performance/chembl.csv"
)

CALIBRATION_N = 10_000
BATCH_SIZE = 128
EPOCHS = 1
RANDOM_SEED = 42


# --------------------------------------------------
# Environment information
# --------------------------------------------------

print("TensorFlow:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
print("Visible devices:", tf.config.get_visible_devices())

# --------------------------------------------------
# Read and clean ChEMBL data
# --------------------------------------------------

print("\nReading ChEMBL data...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    usecols=["Smiles", "AlogP"],
)

df = (
    df
    .dropna(subset=["Smiles", "AlogP"])
    .reset_index(drop=True)
)

print("Clean observations:", len(df))


# --------------------------------------------------
# Select reproducible calibration sample
# --------------------------------------------------

calibration_df = (
    df
    .sample(
        n=CALIBRATION_N,
        random_state=RANDOM_SEED,
    )
    .reset_index(drop=True)
)

print("Calibration observations:", len(calibration_df))
# --------------------------------------------------
# Character vocabulary
# Reproduce the historical full-data encoding
# --------------------------------------------------

all_characters = sorted(
    set("".join(df["Smiles"].astype(str)))
)

char_to_int = {
    char: index + 1
    for index, char in enumerate(all_characters)
}

VOCAB_SIZE = len(char_to_int) + 1
MAX_LENGTH = 404

print("SMILES characters:", len(all_characters))
print("Vocabulary size including padding:", VOCAB_SIZE)
print("Maximum sequence length:", MAX_LENGTH)


# --------------------------------------------------
# Encode calibration SMILES
# --------------------------------------------------

encoded_smiles = [
    [char_to_int[char] for char in smiles]
    for smiles in calibration_df["Smiles"]
]

X = pad_sequences(
    encoded_smiles,
    maxlen=MAX_LENGTH,
    padding="post",
)

y = calibration_df["AlogP"].to_numpy()

print("Encoded X shape:", X.shape)
print("Target y shape:", y.shape)

# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# --------------------------------------------------
# Standardize AlogP target
# --------------------------------------------------

scaler = StandardScaler()

y_train_scaled = scaler.fit_transform(
    y_train.reshape(-1, 1)
).flatten()

y_test_scaled = scaler.transform(
    y_test.reshape(-1, 1)
).flatten()

print("Training target mean:", y_train_scaled.mean())
print("Training target std:", y_train_scaled.std())

# --------------------------------------------------
# Build historical DAT112-derived model
# --------------------------------------------------

tf.keras.utils.set_random_seed(RANDOM_SEED)

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
    ),

    Bidirectional(
        LSTM(
            256,
            return_sequences=True,
        )
    ),

    Dropout(0.2),

    Bidirectional(
        LSTM(256)
    ),

    Dropout(0.2),

    Dense(1),
])

optimizer = Adam(
    learning_rate=1e-4,
    clipnorm=1.0,
)

model.compile(
    optimizer=optimizer,
    loss="mse",
)

model.build(input_shape=(None, MAX_LENGTH))

model.summary()

# --------------------------------------------------
# Metal calibration: one training epoch
# --------------------------------------------------

DEVICE = "/GPU:0"

print("\nCalibration device:", DEVICE)
print("Starting one training epoch...")

start_time = time.perf_counter()

with tf.device(DEVICE):
    history = model.fit(
        X_train,
        y_train_scaled,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.10,
        verbose=1,
    )

end_time = time.perf_counter()

training_seconds = end_time - start_time

training_samples = int(len(X_train) * 0.90)
throughput = training_samples / training_seconds

print("\n--- Metal CALIBRATION RESULTS ---")
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Training samples: {training_samples}")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
