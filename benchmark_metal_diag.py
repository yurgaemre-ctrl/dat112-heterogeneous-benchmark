import time
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


# ==================================================
# Configuration
# ==================================================

DATA_PATH = "data/chembl.csv"
SAMPLE_SIZE = 100_000
RANDOM_SEED = 42
MAX_LENGTH = 404
BATCH_SIZE = 128
WARMUP_EPOCHS = 1
TIMED_EPOCHS = 3


# ==================================================
# Environment
# ==================================================

print("TensorFlow:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())


# ==================================================
# Read and clean ChEMBL
# ==================================================

print("\nReading ChEMBL data...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    usecols=["Smiles", "AlogP"],
)

df = df.dropna(subset=["Smiles", "AlogP"]).copy()

print("Clean observations:", len(df))


# ==================================================
# Fixed benchmark sample
# ==================================================

benchmark_df = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_SEED,
).copy()

print("Benchmark observations:", len(benchmark_df))


# ==================================================
# Vocabulary from FULL cleaned dataset
# ==================================================

all_characters = sorted(
    set(
        char
        for smiles in df["Smiles"]
        for char in smiles
    )
)

char_to_int = {
    char: index + 1
    for index, char in enumerate(all_characters)
}

VOCAB_SIZE = len(char_to_int) + 1

print("SMILES characters:", len(all_characters))
print("Vocabulary size including padding:", VOCAB_SIZE)
print("Maximum sequence length:", MAX_LENGTH)


# ==================================================
# Encode SMILES
# ==================================================

encoded_smiles = [
    [char_to_int[char] for char in smiles]
    for smiles in benchmark_df["Smiles"]
]

X = pad_sequences(
    encoded_smiles,
    maxlen=MAX_LENGTH,
    padding="post",
)

y = benchmark_df["AlogP"].to_numpy()

print("Encoded X shape:", X.shape)
print("Target y shape:", y.shape)


# ==================================================
# Train/test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

scaler = StandardScaler()

y_train_scaled = scaler.fit_transform(
    y_train.reshape(-1, 1)
).flatten()

y_test_scaled = scaler.transform(
    y_test.reshape(-1, 1)
).flatten()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Training target mean:", y_train_scaled.mean())
print("Training target std:", y_train_scaled.std())


# ==================================================
# Reproducibility fingerprints
# ==================================================

def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()

print("\nReproducibility fingerprints:")
print("X_train:", array_hash(X_train))
print("X_test:", array_hash(X_test))
print("y_train:", array_hash(y_train))
print("y_test:", array_hash(y_test))


# ==================================================
# Model
# ==================================================

tf.keras.utils.set_random_seed(RANDOM_SEED)

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128,
    ),
    Bidirectional(
        LSTM(256, return_sequences=True)
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

# Stop immediately if training produces NaN or Inf
nan_callback = tf.keras.callbacks.TerminateOnNaN()

model.build(input_shape=(None, MAX_LENGTH))
model.summary()


# ==================================================
# Warm-up epoch — deliberately NOT timed
# ==================================================

print("\nWARM-UP EPOCH — not included in benchmark")

model.fit(
    X_train,
    y_train_scaled,
    batch_size=BATCH_SIZE,
    epochs=WARMUP_EPOCHS,
    validation_split=0.10,
    callbacks=[nan_callback],
    verbose=1,
)


# ==================================================
# Timed epochs
# ==================================================

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
        callbacks=[nan_callback],
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


# ==================================================
# Benchmark summary
# ==================================================

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

print(f"\nMean time: {np.mean(epoch_times):.2f} s")
print(f"Median time: {np.median(epoch_times):.2f} s")

print(
    f"Mean throughput: "
    f"{np.mean(throughputs):.2f} samples/s"
)

print(
    f"Median throughput: "
    f"{np.median(throughputs):.2f} samples/s"
)

print("========================================")
