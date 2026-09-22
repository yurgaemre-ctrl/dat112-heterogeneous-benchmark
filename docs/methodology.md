# Methodology

## Objective

This project examines how the same neural-network workload behaves across heterogeneous computing hardware, including CPU, Apple Metal, and NVIDIA CUDA devices.

The machine-learning task predicts ChEMBL AlogP values from SMILES molecular representations using a character-level bidirectional LSTM (BiLSTM).

## Dataset

The source dataset is ChEMBL.

For this project, a fixed reference copy of the ChEMBL CSV was created and used across hardware environments. The reference file has SHA-256:

f81721223fb64e0432b89dfca87abbf5b9b6ab246700f1a365dc0667fdf5041e

The dataset itself is not stored in this repository.

After removal of observations with missing or invalid SMILES/AlogP values, the 2026 benchmark contained 2,677,485 eligible observations.

## Molecular representation

SMILES strings are treated as sequences of characters.

Each unique character is mapped to an integer and passed to an embedding layer. Because neural-network batches require a common tensor shape, sequences in the final standardized benchmark are padded or truncated to 404 characters.

The fixed length of 404 is important for hardware comparisons: recurrent neural-network computational cost depends strongly on sequence length. Allowing each hardware run to use a different maximum sequence length would change the workload being benchmarked.

## Neural network

The standardized model contains:

- Embedding layer: 128 dimensions
- Bidirectional LSTM: 256 units, returning sequences
- Dropout: 0.20
- Bidirectional LSTM: 256 units
- Dropout: 0.20
- Dense scalar output

Total trainable parameters: 2,369,921.

Bidirectionality means that the network processes the molecular character sequence in both forward and reverse directions. This allows the learned representation to incorporate context from both sides of a position in the SMILES sequence.

The target variable is AlogP. The training target is standardized using the training-set mean and standard deviation.

Optimization uses Adam with:

- learning rate: 1e-4
- clipnorm: 1.0
- loss: mean squared error

## 10K calibration experiment

An initial calibration used 10,000 observations and one timed epoch to establish the broad performance differences among available hardware.

Measured hardware included:

- Apple M4 CPU
- Apple M4 using Metal acceleration
- NVIDIA RTX 3080

These results are retained as calibration measurements and should not be treated as identical to the later standardized 100K benchmark.

## 2026 standardized 100K benchmark

The final controlled benchmark uses:

- deterministic sample: 100,000 observations
- random seed: 42
- train/test split: 80/20
- fixed sequence length: 404
- batch size: 128
- one warm-up epoch
- three timed epochs
- validation_split: 0.10

Because validation_split reserves 10% of the 80,000 training observations during fitting, each timed epoch trains on 72,000 observations. Throughput is therefore calculated as:

72,000 / elapsed epoch time.

For the final T4 and A100 runs, the encoded arrays were verified using SHA-256 fingerprints:

X_train:
37aac87222e853bb5a7f6eca258043e966f15a2a9342a96ba7087688d85c0b2c

X_test:
b76d5974c68d85adac7699f40f2fd593ed55a1409000be080e22917a8cf97617

y_train:
2a3e52c3227351d161f7a55b5d9ec27ffd03cec373d8a01cac5987598aa6dc69

y_test:
21301ff3c54b5ed186e40d6df58c4816e9943c9f82c80be6bd70a7390a762e23

Matching fingerprints demonstrate that the T4 and A100 benchmark runs used identical encoded inputs and targets.

## Final controlled GPU results

NVIDIA Tesla T4:

- mean epoch time: 180.34 seconds
- mean throughput: 399.24 samples/second

NVIDIA A100-SXM4-40GB:

- mean epoch time: 43.06 seconds
- mean throughput: 1,672.22 samples/second

Under this workload, the A100 therefore achieved approximately 4.19 times the T4 throughput.

## 2025 full-data A100 experiment

A separate earlier experiment trained an A100 model using the much larger full-data workload:

- X_train: 2,141,988 x 404
- X_test: 535,497 x 404
- 15 training epochs
- approximately 1,154-1,157 seconds per epoch
- final scaled training MSE: 0.0019
- original-scale test MSE: 0.0034

This experiment demonstrates full-scale model training rather than controlled cross-hardware benchmarking. It is retained for historical comparison but is not directly interchangeable with the 2026 100K benchmark.

## Metal diagnostic

Apple Metal testing also exposed a numerical-stability/NaN episode during development. This is retained as part of the experimental record rather than omitted.

The episode illustrates a broader point of heterogeneous computing: equivalent high-level TensorFlow/Keras models can exhibit different numerical or runtime behaviour across CPU, Metal, and CUDA software/hardware stacks.

## Reproducibility

The repository separates:

1. source and diagnostic code;
2. executed notebook evidence;
3. machine-readable benchmark results;
4. dataset and tensor fingerprints; and
5. methodological documentation.

Large ChEMBL source files, virtual environments, caches, and trained-model artifacts are intentionally excluded from Git.
