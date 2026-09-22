# ============================================================
# ChEMBL Heterogeneous Hardware Benchmark
# Reference 100K Benchmark Protocol
# ============================================================

import time
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/chembl.csv"

RANDOM_SEED = 42
BENCHMARK_SIZE = 100_000
BATCH_SIZE = 128
WARMUP_EPOCHS = 1
TIMED_EPOCHS = 3

# Fixed sequence length for cross-hardware comparability
MAX_LENGTH = 404


# ============================================================
# Environment
# ============================================================

print("TensorFlow:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ============================================================
# Read ChEMBL
# ============================================================

print("\nReading ChEMBL data...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    usecols=["Smiles", "AlogP"],
    low_memory=False,
)

df = df.dropna(subset=["Smiles", "AlogP"]).copy()

df["Smiles"] = df["Smiles"].astype(str)
df["AlogP"] = pd.to_numeric(df["AlogP"], errors="coerce")

df = df.dropna(subset=["AlogP"]).copy()

print("Clean observations:", len(df))


# ============================================================
# Deterministic benchmark sample
# ============================================================

df = df.sample(
    n=BENCHMARK_SIZE,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

print("Benchmark observations:", len(df))


# ============================================================
# Character vocabulary
# ============================================================

smiles = df["Smiles"].tolist()

characters = sorted(set("".join(smiles)))

char_to_int = {
    char: index + 1
    for index, char in enumerate(characters)
}

VOCAB_SIZE = len(char_to_int) + 1

print("SMILES characters:", len(characters))
print("Vocabulary size including padding:", VOCAB_SIZE)
print("Maximum sequence length:", MAX_LENGTH)


# ============================================================
# Encode SMILES
# ============================================================

encoded_smiles = [
    [char_to_int[char] for char in smile]
    for smile in smiles
]

X = pad_sequences(
    encoded_smiles,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post",
)

y = df["AlogP"].to_numpy(dtype=np.float64)

print("Encoded X shape:", X.shape)
print("Target y shape:", y.shape)


# ============================================================
# Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# Standardize target
# ============================================================

scaler = StandardScaler()

y_train_scaled = scaler.fit_transform(
    y_train.reshape(-1, 1)
).flatten()

y_test_scaled = scaler.transform(
    y_test.reshape(-1, 1)
).flatten()

print("Training target mean:", y_train_scaled.mean())
print("Training target std:", y_train_scaled.std())


# ============================================================
# Reproducibility fingerprints
# ============================================================

def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


print("\nReproducibility fingerprints:")
print("X_train:", array_hash(X_train))
print("X_test:", array_hash(X_test))
print("y_train:", array_hash(y_train))
print("y_test:", array_hash(y_test))


# ============================================================
# Model
# ============================================================

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

model.compile(
    optimizer=Adam(
        learning_rate=1e-4,
        clipnorm=1.0,
    ),
    loss="mse",
)

model.build(input_shape=(None, MAX_LENGTH))

model.summary()


# ============================================================
# Warm-up epoch
# ============================================================

print("\nWARM-UP EPOCH — not included in benchmark")

model.fit(
    X_train,
    y_train_scaled,
    batch_size=BATCH_SIZE,
    epochs=WARMUP_EPOCHS,
    validation_split=0.10,
    verbose=1,
)


# ============================================================
# Timed epochs
# ============================================================

training_samples = int(len(X_train) * 0.90)

epoch_times = []
throughputs = []

print("\nSTARTING TIMED BENCHMARK")

for epoch in range(TIMED_EPOCHS):

    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train_scaled,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_split=0.10,
        verbose=1,
    )

    elapsed = time.perf_counter() - start_time

    throughput = training_samples / elapsed

    epoch_times.append(elapsed)
    throughputs.append(throughput)

    print(
        f"\nTimed epoch {epoch + 1}: "
        f"{elapsed:.2f} s | "
        f"{throughput:.2f} samples/s | "
        f"loss={history.history['loss'][-1]:.6f} | "
        f"val_loss={history.history['val_loss'][-1]:.6f}"
    )


# ============================================================
# Final benchmark summary
# ============================================================

mean_time = np.mean(epoch_times)
median_time = np.median(epoch_times)

mean_throughput = np.mean(throughputs)
median_throughput = np.median(throughputs)

print("\n========================================")
print("BENCHMARK RESULTS")
print("========================================")

for i, (elapsed, throughput) in enumerate(
    zip(epoch_times, throughputs),
    start=1,
):
    print(
        f"Epoch {i}: "
        f"{elapsed:.2f} s | "
        f"{throughput:.2f} samples/s"
    )

print()
print(f"Mean time: {mean_time:.2f} s")
print(f"Median time: {median_time:.2f} s")
print(f"Mean throughput: {mean_throughput:.2f} samples/s")
print(f"Median throughput: {median_throughput:.2f} samples/s")
print("========================================")