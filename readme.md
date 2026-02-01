# Health Monitoring Toolbox: For Networked Systems

## Project Overview

![architecture](health_monitoring_framework.png)

This is a comprehensive toolbox designed for health monitoring faults in networked systems. The toolbox integrates three interconnected machine learning pipelines:

1. **Fault Detection (i.e. Module-level health monitoring)** - Identifies whether a anomaly exists in the module-level.
2. **Topology Estimation (i.e. System-level health monitoring)** - Identifies if there is a system-level anomaly by learning the underlying connections between system modules.
3. **Feature Extraction** - Extracts meaningful features from raw sensor data
4. **Data Pipeline** - Handles data loading, preprocessing, and augmentation from multiple datasets

This is a research thesis project that provides end-to-end health monitoring capabilities on module and system level across multiple industrial and academic datasets.

---

## Core blocks

### 1. **Fault Detection Block (Module-level Health Monitoring)** (`fault_detection/`)

**Purpose:** Detects whether each module in system is operating healthy or unhealthy.

**Key Components:**
- **`detector.py`** - Main fault detection class using multiple anomaly detection algorithms:
  - **Isolation Forest (IF)** - Isolation-based anomaly detection
  - **One-Class SVM** - Support Vector Machine for anomaly detection
  - **SVC** - Support Vector Classifier for binary classification
  
- **`train.py`** - Training pipeline for fault detection models
- **`multi_train.py`** - Imports `train.py` to perform batch training for multiple fault detection models (can perform hyperparameter sweeps)
- **`infer.py`** - Single model inference pipeline
- **`multi_infer.py`** - Batch inference across multiple trained models by importing `infer.py`.
- **`settings/`** - Configuration management for training and inference. The doctring in the config files explains what each metric mean. Users must use modules with name `*_config.py` to alter the settings for model training and inference. 

**Workflow:**
```
Raw Data → Feature Extraction → Anomaly Detection Model → Fault/No-Fault Classification
```

**Output:** Binary predictions (fault or normal) with confidence scores and ROC/F1 metrics.

---

### 2. **Topology Estimation Block (System-level Health Monitoring)** (`topology_estimation/`)

**Purpose:** Estimates the underlying graph structure (which components are connected) from multivariate time series data using Neural Relational Inference (NRI).

**Key Components:**
- **`nri.py`** - NRI model implementation (PyTorch Lightning):
  - Encoder: Learns latent graph representations from data
  - Decoder: Reconstructs trajectories given the learned graph
  - Loss functions: KL divergence, MSE loss, Cross Entropy Loss.
  - Schedulers: Beta annealing and temperature scheduling for training
  
- **`encoder.py`** - Contains the graph neural network code in form of Message Passing class and the Encoder class. It outputs the topology infered from the node trajectories.
- **`decoder.py`** - Contains the decoder code that also uses graph neural netowrk framework along with Gated Recurrent Unit (GRU) to reconstruct the node trajectories from the infered topology. 
- **`train.py`** - Training loop with PyTorch Lightning. Used for single model training.
- **`multi_train.py`** - Imports `train.py` to run batch model training with which hyperparameter sweeps can be performed. 
- **`infer.py` / `multi_infer.py`** - Model inference interfaces. Similar to the training, `infer.py` is used for infering a single model. `multi_infer.py` is used to infer multiple trained models, either sequentially or parallelly.  
- **`utils/`** - Custom loss functions, schedulers, and PyTorch models like MLP and GRU.  
- **`settings/`** - Training and inference configuration management. The use of each metric is explained as docstring in the config files. Users must use modules with name `*_config.py` to alter the settings for model training and inference. 

**Workflow:**
```
Time Series Data → Feature extraction → NRI Encoder → Latent Graph Representation → Decoder → Node trajectories
```

**Output:** Estimated graph adjacency matrices showing component relationships with edge probabilities.

---

### 3. **Feature Extraction Block** (`feature_extraction/`)

**Purpose:** Extracts both time-domain and frequency-domain features from raw sensor data.

### New feature ranking modules (used for the Thesis)

**Key Components:**
- **`extractor.py`** - Main feature extraction pipeline:
  - `TimeFeatureExtractor` - Extracts time-domain features (mean, std, RMS, kurtosis, etc.)
  - `FrequencyFeatureExtractor` - Extracts frequency-domain features (FFT, spectral moments)
  - `FeatureReducer` - Dimensionality reduction using PCA

- **`tf.py`** - Time features implementations
- **`ff.py`** - Frequency features implementations
- **`selector.py`** - Feature selection and ranking algorithms which uses LDA or PCA based feature ranking.

**Feature Categories:**
- **Time Domain:** (37 time features)Mean, Std Dev, RMS, Peak, Crest Factor, Kurtosis, Skewness, Peak-to-Peak etc.
- **Frequency Domain:** FFT magnitudes, Power Spectral Density, Spectral Centroid, Spectral Spread etc.

**Note**: During the time of thesis, only the FFT magnitudes and Power Spectral Density worked. Some of the other frequency features produced a null value. These need to be fixed for future studies. 

**Output:** High-quality feature vectors ready for module-level and system-level health monitoring.

### Old feature ranking modules (not used for the Thesis)
- **`performance.py`** - Feature performance evaluation metrics
- **`ranking_utils/`** - Advanced feature ranking:
  - `feature_performance.py` - Statistical performance metrics
  - `feature_robustness.py` - Robustness analysis across conditions
  - `frequency_features.py`, `time_features.py` - Domain-specific extractors
  - `ranking_algorithms.py` - Multiple ranking algorithms (correlation, mutual info, etc.)

- **`lucas/`** - Original feature extraction algorithms by Lucas from ASML intership.
- **`settings/`** - Feature extraction configuration

---

### 4. **Data loading and preprocessing** (`data/`)

**Purpose:** Manages data loading, preprocessing, augmentation, and dataset handling.

**Key Components:**
- **`prep.py`** - Data preprocessing pipeline:
  - `DataPreprocessor` - Main pipeline class
  - Handles segmentation, augmentation, and train/val/test splitting
  - Supports both fault detection and topology estimation labeling

- **`transform.py`** - Data transformation utilities:
  - `DomainTransformer` - Converts between time/frequency domains
  - `DataNormalizer` - Normalization and scaling

- **`augment.py`** - Data augmentation techniques
- **`config.py`** - Data configuration management
- **`faults.py`** - Fault type definitions and mappings

**Supported Datasets:**
- **Bearing (CWRU)** - Case Western Reserve University bearing dataset
- **Mass-Spring-Damper** - Mass Spring Damper system simulated in MATLAB.

**Data Features:**
- Multi-node sensor measurements
- Node-level labels (for fault detection)
- Edge-level labels (for topology estimation)
- Automatic train/val/test splitting
- Data augmentation pipelines

### Mass-Spring-Damper (MSD) system in MATLAB

The generalized_msd_2 code offers you to simulate the
dynamics of multiple masses connected via springs and dampers. The motion
is simulated in 1 dimension along the horizontal axis. 
With this toolbox, you can make various configurations of the MSD
network. The picture "msd_network_schematic" attached with these code files provides a pictorial represeantion of the various connection you can make.

![architecture](data/matlab_scripts/msd_scripts/msd_network_schematic_example.png)

Based on the number of masses in the system, you make a machine. 
The configuration of the topology can vary between healthy and unhealthy.
There are two files to alter the system. These can be found in `.../machines/<machine_name>/<healthy or unhealthy>/<topology name>/`
`config_machine_param.m` and `generate_machine`.

- `generate_machine.m`: design the connections by adding which pairs of
masses you need the spring and damper connected. This is added in
conn.pairs varaible.
- `config_machine_param.m`: design the value of the mass, spring and
damper. There are two types of springs and dampers. (i) The ones that
connect two masses (denoted by just spring_k and damper_d) and (ii) The
ones that connect mass to a wall (wall is a immovable object with
infinite mass). These are denoted by k_wall_lin, d_wall_lin
respectively. 

#### How to run the MSD simulation
Type the commands in MATLAB comand window.

**Step 1**: Use addpath in MATLAB to add the msd_scripts directory to the MATLAB environement. 
`addpath '<your previous folders>\AFD_thesis\data\matlab_scripts\msd_scripts'`

**Step 2**: Use generate_dataset to make the MSD dataset that runs the run_dynamics and stores its results into the datasets folder. 
`generate_dataset(<machine_name>, <scenario_name>, <healthy or unhealthy>, <topology_name>, <serial_num>)`

---

## Single v/s Multi-models

Both fault detection and topology estimation models have two types of modes - single mode and multi-mode

- Single mode is when you wish to train, custom test and infer just one model. This mode allows you to ensure the model code runs appropiately and do quick checks on some hyperparameters.

- Multi mode is when you wish to train, custom test and infer multiple models. There are two ways to run this mode. 

**Series** - Each of the multiple models are run sequentially. This is useful if your PC does not have multiple CPU or GPU cores (GPU is used for topology estimation model).
**Parallel**  - Each of the multiple models are run parallely across the multi-CPU cores. If you choose this option for topology estimation model, the CPU cores will be used instead of GPU as the PC used to develop the toolbox has one GPU. Future collaborators can modify this functionality to switch from CPU to GPU if multiple GPUs are possible. 

## How to select a trained model
Run the infer_config.py file in settings folders of fault_detection and topology_estimation. Follow the instructions provided in the terminal to select your desired trained model. 

## 🚀 Quick Start: Command Line Interface

The following commands must be run in the root directory of AFD folder
`...\AFD`:

### Fault detection (Module-level health monitoring)
```bash
# Training
# train a single fault detection model
python -m fault_detection.train

# train a batch of fault detection model
python -m fault_detection.multi_train --parallel

# Custom Testing
# test a trained model
python -m fault_detection.infer --run-type custom_test

# test a batch of trained model
python -m fault_detection.multi_infer --run-type custom_test --parallel

# Inference/Prediction
# run inference on new data
python -m fault_detection.infer --run-type predict

# run inference on batch of new data for each model of module
python -m fault_detection.multi_infer --run-type predict --parallel
```

### Topology Estimation (System-level health monitoring)
```bash
# Training
# train a single topology estimation model
python -m topology_estimation.train --framework nri

# train a batch of topology estimation model
python -m topology_estimation.multi_train --framework nri --parallel

# Custom Testing
# test a trained model
python -m topology_estimation.infer --framework nri --run-type custom_test

# test a batch of trained model
python -m topology_estimation.multi_infer --framework nri --run-type custom_test --parallel

# Inference/Prediction
# run inference on new data
python -m topology_estimation.infer --framework nri --run-type predict

# run inference on batch of new data for each model of module
python -m topology_estimation.multi_infer --framework nri --run-type predict --parallel
```

### Configuration
Each module has dedicated configuration files in `settings/`:
- `train_config.py` - Training hyperparameters and model settings
- `infer_config.py` - Inference settings and model selection
- `feature_config.py` - Feature extraction parameters
- `rank_config.py` - Feature ranking parameters

---

## 📊 Model Selection & Management

### Settings Management
Each module includes a `settings/manager.py` that:
- Loads and saves configurations
- Manages model versioning
- Tracks training history and metrics
- Handles hyperparameter experiments

### Logging & Monitoring
- **TensorBoard Integration** - Real-time training visualization
- **CSV Logging** - Performance metrics at each epoch
- **Model Checkpoints** - Automatic model saving during training
- **Sweep Logs** - Comprehensive hyperparameter sweep documentation

---

## 📂 Directory Organization

```
AFD_thesis/
├── fault_detection/          # Fault detection pipeline
│   ├── detector.py
│   ├── train.py, infer.py
│   ├── multi_train.py, multi_infer.py
│   ├── settings/
│   ├── logs/                 # Training logs by dataset
│   └── docs/
│
├── topology_estimation/       # Graph structure learning (NRI)
│   ├── nri.py, encoder.py, decoder.py
│   ├── train.py, infer.py
│   ├── multi_train.py, multi_infer.py
│   ├── utils/               # Loss functions, schedulers, models
│   ├── settings/
│   ├── logs/
│   └── docs/
│
├── feature_extraction/        # Feature engineering
│   ├── extractor.py, ff.py, tf.py
│   ├── selector.py, performance.py
│   ├── ranking_utils/        # Advanced ranking algorithms
│   ├── lucas/                
│   ├── settings/
│   └── logs/
│
├── data/                      # Data pipeline
│   ├── prep.py              # Preprocessing pipeline
│   ├── transform.py          # Data transformations
│   ├── augment.py            # Data augmentation
│   ├── config.py, faults.py
│   └── datasets/
│       ├── bearing/          # CWRU bearing dataset
│       ├── mass_sp_dm/       # Mass-spring-damper
│       └── spring_particles/ # Multi-particle systems
│
└── afd_env.yml               # Conda environment specification
```

---

## 🛠️ Technologies & Dependencies

**Core Libraries:**
- **PyTorch** - Deep learning framework
- **PyTorch Lightning** - Training infrastructure
- **Scikit-Learn** - Classical ML algorithms (Isolation Forest, SVM)
- **NumPy & SciPy** - Numerical computing and signal processing
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

**Environment:**
- Python 3.x
- CUDA (optional, for GPU acceleration)
- Conda (for environment management)

Install environment:
```bash
conda env create -f afd_env.yml
conda activate afd_env
```

---

## 📈 Key Features

### ✅ Multi-Dataset Support
- Seamlessly switch between CWRU Bearing and Mass-Spring-Damper

### ✅ Multiple Anomaly Detection Algorithms
- Isolation Forest
- One-Class SVM
- Support Vector Classification
- Hyperparameter optimization and comparison

### ✅ Advanced Feature Engineering
- 37+ time and frequency domain features
- Automatic feature selection and ranking

### ✅ Graph Structure Learning
- Neural Relational Inference (NRI) for discovering system topology
- Interpretable graph representations

### ✅ Comprehensive Logging & Monitoring
- TensorBoard integration for real-time training visualization
- Detailed metric tracking
- Hyperparameter sweep management

### ✅ Batch Processing
- Train/test multiple models in parallel
- Hyperparameter grid search
- Cross-dataset evaluation

---

**Last Updated:** January 2026
