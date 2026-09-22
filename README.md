# ChEMBL Heterogeneous Hardware Benchmark

A reproducible experiment comparing neural-network training performance across heterogeneous computing hardware using a character-level bidirectional LSTM and molecular data from ChEMBL.

## Research question

**How differently does the same neural-network workload perform when moved across CPU, Apple Metal, and NVIDIA CUDA hardware?**

The machine-learning task predicts molecular AlogP values from SMILES strings. The scientific prediction task provides a realistic recurrent neural-network workload; the primary focus here is computational performance and reproducibility across hardware environments.

The project developed through three stages:

1. 10,000-observation hardware calibration;
2. standardized 100,000-observation benchmark; and
3. comparison with an earlier full-data A100 training experiment.

## Neural-network workload

SMILES represents molecular structures as character sequences. The benchmark converts each SMILES string into an integer sequence and processes it through:

    SMILES
      |
      v
    character encoding
      |
      v
    pad/truncate to 404 positions
      |
      v
    Embedding (128 dimensions)
      |
      v
    Bidirectional LSTM (256 units)
      |
      v
    Dropout (0.20)
      |
      v
    Bidirectional LSTM (256 units)
      |
      v
    Dropout (0.20)
      |
      v
    Dense output
      |
      v
    predicted AlogP

The model contains **2,369,921 trainable parameters**.

### Why 404 positions?

SMILES strings have different lengths, while batched neural-network computation requires tensors with consistent dimensions.

The standardized benchmark therefore fixes sequence length at **404 characters**. Shorter strings are padded and longer strings are truncated.

This is also necessary for hardware comparability. Recurrent-network computational cost depends strongly on sequence length, so changing the sequence length would change the workload being benchmarked.

### Why bidirectional?

A bidirectional LSTM processes each sequence in both directions. One recurrent pass reads forward and another backward.

The resulting representation can therefore incorporate context from both directions of the molecular character sequence rather than relying only on preceding characters.

## Data reproducibility

For this project, a fixed reference copy of the ChEMBL CSV was created and used across hardware environments.

SHA-256:

    f81721223fb64e0432b89dfca87abbf5b9b6ab246700f1a365dc0667fdf5041e

The large ChEMBL source file is intentionally not stored in this repository.

The 2026 benchmark produced **2,677,485 eligible observations** after preprocessing.

## 10K hardware calibration

| Hardware | Epoch time | Throughput |
| --- | ---: | ---: |
| Apple M4 CPU | 133.87 s | 53.78 samples/s |
| Apple M4 Metal | 36.05 s | 199.72 samples/s |
| NVIDIA RTX 3080 | 11.84 s | 607.98 samples/s |

These measurements established the broad performance range across CPU, Metal, and CUDA hardware. They are retained as calibration results rather than treated as identical to the subsequent standardized benchmark.

## Final 100K controlled benchmark

The controlled benchmark uses:

- 100,000 observations
- random seed 42
- 80/20 train/test split
- fixed sequence length 404
- batch size 128
- one warm-up epoch
- three timed epochs
- 10% validation split within the training data
- identical BiLSTM architecture

Because the validation split reserves 10% of the 80,000 training observations, each timed epoch trains on **72,000 observations**.

### Results

| Hardware | Epoch 1 | Epoch 2 | Epoch 3 | Mean epoch | Mean throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA Tesla T4 | 180.20 s | 180.47 s | 180.36 s | **180.34 s** | **399.24 samples/s** |
| NVIDIA A100-SXM4-40GB | 43.10 s | 43.03 s | 43.04 s | **43.06 s** | **1,672.22 samples/s** |

Under this controlled workload, the A100 delivered approximately **4.19x the throughput of the T4**.

## Reproducibility fingerprints

The final T4 and A100 runs generated identical encoded-data fingerprints:

    X_train
    37aac87222e853bb5a7f6eca258043e966f15a2a9342a96ba7087688d85c0b2c

    X_test
    b76d5974c68d85adac7699f40f2fd593ed55a1409000be080e22917a8cf97617

    y_train
    2a3e52c3227351d161f7a55b5d9ec27ffd03cec373d8a01cac5987598aa6dc69

    y_test
    21301ff3c54b5ed186e40d6df58c4816e9943c9f82c80be6bd70a7390a762e23

These hashes provide evidence that the controlled T4 and A100 runs operated on identical encoded inputs and targets.

## Earlier full-data A100 experiment

An earlier A100 experiment used a substantially larger workload:

- X_train: 2,141,988 x 404
- X_test: 535,497 x 404
- 15 training epochs
- approximately 1,154-1,157 seconds per epoch
- final scaled training MSE: 0.0019
- original-scale test MSE: 0.0034

This is retained as evidence of full-scale training, but it is **not treated as directly comparable** with the standardized 100K hardware benchmark.

## Heterogeneous computing is more than speed

Apple Metal testing also exposed a numerical-stability/NaN episode during development.

That observation is retained in the experimental record. Hardware portability involves more than elapsed time: software backends, numerical implementations, kernels, and accelerator architectures can affect runtime behaviour even when the high-level TensorFlow/Keras model appears equivalent.

## Repository structure

    .
    ├── README.md
    ├── benchmark.py
    ├── benchmark_cpu.py
    ├── benchmark_metal_diag.py
    ├── calibrate.py
    ├── smoke_test.py
    ├── docs/
    │   └── methodology.md
    ├── notebooks/
    │   └── chembl_colab_t4_a100_2026.ipynb
    └── results/
        ├── benchmark_results.csv
        ├── calibration_results.csv
        ├── a100_full_data_2025.csv
        └── a100_full_data_2025_notes.txt

## Reproducibility

The repository preserves source and diagnostic code, the executed Colab notebook, machine-readable results, dataset SHA-256 verification, encoded tensor fingerprints, and detailed methodological documentation.

Large datasets, virtual environments, caches, and trained-model artifacts are excluded from Git.

See `docs/methodology.md` for the detailed experimental specification.

## License

The original code and documentation in this repository are available under the MIT License. See [LICENSE](LICENSE).

The ChEMBL source dataset is not included in this repository and remains subject to its own terms and licensing conditions.
