# Reinforcement Learning Sustainability Benchmark

A benchmark among various reinforcement learning algorithms to assess the tradeoff between performance and costs in terms of energy and time.

This project was developed as part of the *Software Engineering for Artificial Intelligence* course, therefore, in addition to the code, the repository contains the full documentation produced during the project development (see [final_report.pdf](documentation/final_report/final_report.pdf)).

## Overview

The aim of this project is to evaluate and compare several deep reinforcement learning (DRL) algorithms not only in terms of performance but also regarding their energy consumption and training time. Experiments are performed on a standardized benchmark (Atari 100k) and tracked using tools such as Weights & Biases, TensorBoard, and CodeCarbon. This approach provides insights into the sustainability implications of DRL methods.

*The development and execution environment is based on CleanRL’s work. We also followed their single-file implementation philosophy to ensure simplicity and reproducibility.*

## Installation

### Requirements

- **Python:** Version >=3.7.1,<3.11 (the project was developed using Python 3.10.9)
- **Poetry:** Version 1.2.1+ (installation instructions available on the [Poetry website](https://python-poetry.org/docs/#installation); pipx might be needed)
- **Weights & Biases Account:** Optional, for online experiment tracking

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/lucastrefezza/reinforcement-learning-sustainability-benchmark.git
   ```

2. **Navigate to the Code Directory**

   ```bash
   cd reinforcement-learning-sustainability-benchmark/code
   ```

3. **Install Dependencies**

   Run the following command to install all required packages using Poetry:
   
   ```bash
   poetry install
   ```
   
   > **Note:** Ensure that you have Poetry installed. You may need to use pipx for its installation as detailed on the Poetry installation page.

4. **Install Atari-Specific Dependencies**

   Run the extra installation for Atari environments:
   
   ```bash
   poetry install -E atari
   ```

5. **Weights & Biases Setup (Optional)**

   - Log in to your [Weights & Biases](https://wandb.ai/) account to enable experiment tracking.
   - (Tracking using TensorBoard is always enabled by default.)

6. **Handling CodeCarbon Issues**

   CodeCarbon should be installed along with the project dependencies. However, if you experience any issues, follow these steps:
   
   - Re-install CodeCarbon using pip:
   
     ```bash
     pip install codecarbon
     ```
   
   - Since this might change your PyTorch and CUDA versions, reinstall them with:
   
     ```bash
     poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
     ```

7. **Windows Users: Administrator Privileges**

   If you are running on Windows and want to take advantage of the wandb tracking, launch PowerShell or Command Prompt with administrator privileges. This is necessary for reading and uploading TensorBoard log files correctly, ensuring that Weights & Biases displays complete tracking information (beyond the automatically collected system metrics).

## Usage

After installation, you can start running the experiments by executing the provided training scripts located in the `code` directory. There are two ways to launch a script:

1. **By Opening a Poetry Shell:**

   First, in the `code` directory, start the Poetry shell:
   
   ```bash
   poetry shell
   ```
   
   Then, run the script:
   
   ```bash
   python cleanrl/dqn_atari.py [--track] [--save-model] [--total-timesteps 100000]
   ```
2. **Without Opening a Poetry Shell:**

   ```bash
   poetry run python cleanrl/dqn_atari.py [--track] [--save-model] [--total-timesteps 100000]
   ```

## Documentation

In addition to the code, the repository includes extensive documentation detailing the project methodology, experiment setup, and results. For the complete project report, please refer to [final_report.pdf](documentation/final_report/final_report.pdf).
