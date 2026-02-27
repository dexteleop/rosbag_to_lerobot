# ROS2 Bag to LeRobot Dataset Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![CUDA 12.6+](https://img.shields.io/badge/CUDA-12.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![LeRobot 0.3.3](https://img.shields.io/badge/LeRobot-0.3.3-orange.svg)](https://github.com/huggingface/lerobot)
[![Dataset v2.1](https://img.shields.io/badge/Dataset%20Format-v2.1-orange.svg)](https://github.com/huggingface/lerobot)

A toolkit for converting ROS2 bag files to LeRobot dataset format v2.1.

## 📑 Table of Contents

- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
  - [1. Install Conda](#1-install-conda)
  - [2. Create Conda Environment](#2-create-conda-environment)
  - [3. Install Project-Specific LeRobot](#3-install-project-specific-lerobot)
- [Usage](#-usage)
  - [Scenario 1: Single Rosbag with Multiple Episodes](#scenario-1-single-rosbag-with-multiple-episodes)
  - [Scenario 2: Each Rosbag Contains Single Episode](#scenario-2-each-rosbag-contains-single-episode)
- [Important Notes](#️-important-notes)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 📋 System Requirements

- **OS**: Ubuntu 22.04
- **ROS2 Version**: Humble
- **Python Version**: 3.11
- **CUDA Version**: 12.6+ (tested with 12.8)
- **LeRobot Version**: 0.3.3 (Modified)
- **Dataset Format**: v2.1
- **Conda/Anaconda**: For environment management

## 🚀 Quick Start

### 1. Install Conda

Download and install Miniconda or Anaconda:

```bash
# Install Anaconda (replace with your actual installer name)
chmod +x Anaconda3-2025.06-0-Linux-x86_64.sh
./Anaconda3-2025.06-0-Linux-x86_64.sh
```

After installation, open a new terminal. You should see `(base)` before your username.

### 2. Create Conda Environment

```bash
# Navigate to project directory
cd rosbag_to_lerobot

# Create environment from environment.yml
conda env create -f environment.yml -n rosbag2lerobot

# Activate the environment
conda activate rosbag2lerobot
```

### 3. Install Project-Specific LeRobot

```bash
# Install modified lerobot version from the project
cd lerobot
pip install -e .
cd ..
```

## 📖 Usage

This project provides two conversion methods for different data collection scenarios.

### Scenario 1: Single Rosbag with Multiple Episodes

**Use case**: Original rosbag contains operator start (X key) and end (Y key) markers.

**Script**: `convert_rosbag_with_markers.py`

```bash
# Setup ROS2 environment
# Activate conda environment (ROS2 is already included)
conda activate rosbag2lerobot
unset PYTHONPATH

# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
python scripts/convert_rosbag_with_markers.py \
  --input_directory ./data/rosbags/multiepisode_rosbag \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```

**Parameters**:
- `--input_directory`: Directory containing ROS2 bag files
- `--output`: Output dataset name (format: `username/dataset_name`)
- `--fps`: Target frame rate (default: 30)
- `--task`: Task description
- `--multibag`: (Optional) Use if directory contains multiple rosbag folders
- `--enforce_four_video_topics`: (Optional) Enforce that rosbag must have all 4 video topics

### Scenario 2: Each Rosbag Contains Single Episode

**Use case**: Rosbags are pre-sliced, each containing only one episode.

**Script**: `convert_sliced_rosbags.py`

```bash
# Setup ROS2 environment
# Activate conda environment (ROS2 is already included)
conda activate rosbag2lerobot
unset PYTHONPATH

# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
python scripts/convert_sliced_rosbags.py \
  --input_directory ./data/rosbags/single_episode_segments \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```

## ⚠️ Important Notes

1. **Critical**: Always delete previous dataset cache before re-running conversion scripts
2. Ensure conda environment is activated (`conda activate rosbag2lerobot`) before running scripts. ROS2 is included in the environment.
3. Run `unset PYTHONPATH` to avoid conflicts with system Python packages
4. First run may take longer due to dependency downloads
5. Program output is automatically written to `.log` files. Terminal shows ROS2 system output only.

## 📁 Project Structure

```
rosbag_to_lerobot/
├── scripts/
│   ├── convert_rosbag_with_markers.py   # Convert rosbags with X/Y episode markers
│   ├── convert_sliced_rosbags.py        # Convert pre-sliced rosbag segments
├── lerobot/                             # Modified LeRobot library
├── environment.yml                      # Conda environment specification
├── README.md                            # This file
└── LICENSE                              # MIT License
```

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is based on [LeRobot v0.3.3](https://github.com/huggingface/lerobot) (Apache 2.0 License).
