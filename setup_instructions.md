# Project Setup Instructions (Using Gymnasium and Conda)

This guide details how to set up the environment using Conda, which is the recommended approach for this project, especially on Windows.

## Prerequisites

- Anaconda or Miniconda installed (https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA drivers installed (if using GPU acceleration)

## Step 1: Create and Activate Conda Environment

1.  **Navigate to Project Directory:**
    Open your terminal (Anaconda Prompt or Command Prompt where conda is in PATH) and navigate to the project directory:
    ```bash
    cd C:\Dev\AIClass\AICarRacing
    ```

2.  **Create Environment from YAML:**
    Use the provided `conda_environment.yml` file to create the environment. This installs all necessary packages with the correct versions and dependencies.
    ```bash
    conda env create -f conda_environment.yml
    ```
    *(This might take a few minutes)*

3.  **Activate the Environment:**
    ```bash
    conda activate racing
    ```
    You should see `(racing)` appear at the beginning of your terminal prompt.

## Step 2: Verify Installation

1.  **Check Core Libraries:**
    Run this command to verify Python, Gymnasium, PyTorch, and CUDA availability:
    ```bash
    python -c "import sys; import gymnasium as gym; import torch; print(f'Python: {sys.version.split()[0]}'); print(f'Gymnasium: {gym.__version__}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
    ```

2.  **Test CarRacing Environment:**
    Run the test script to ensure the CarRacing environment loads and runs:
    ```bash
    python test_carracing.py
    ```
    You should see a window pop up showing the game being played with random actions.

## Using the Environment

- Always activate the environment before running any project scripts:
  ```bash
  conda activate racing
  ```
- To deactivate when finished:
  ```bash
  conda deactivate
  ```

## Troubleshooting

-   **`conda env create` fails:** Ensure you have the latest version of conda (`conda update conda`) and try again. Check for conflicting packages or network issues.
-   **CUDA not available:** Verify your NVIDIA drivers are up-to-date and that the `pytorch-cuda` version in `conda_environment.yml` matches your driver/GPU capabilities. Check `nvidia-smi` output.
-   **CarRacing environment errors:** Ensure `gymnasium[box2d]` installed correctly. Sometimes reinstalling helps: `pip uninstall gymnasium pyglet pygame box2d-py -y && pip install gymnasium[box2d]`. 