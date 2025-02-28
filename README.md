README.md

# SHA-3 CUDA Implementation

This repository contains a raw CUDA implementation of SHA-3 (Keccak-f[1600]) based on Choi and Seo’s "Fast Implementation of SHA-3 in GPU Environment." It processes 1,048,576 messages (~2.41 GB input) in parallel on an NVIDIA GPU, targeting the paper’s single-stream throughput of 88.51 Gb/s. The code includes basic throughput measurement and runs on both WSL (Windows Subsystem for Linux) and Ubuntu Linux environments.

## Features
- **θ Optimization**: PTX inline assembly (Algorithm 2)—faster XOR operations.
- **ρ+π**: Direct indexing—no π table (Section III.D)—optimized for GPU.
- **Coalesced Memory**: Column-wise input layout (Section III.E)—enhances GPU memory access.
- **Single-Stream**: Implements paper’s 88.51 Gb/s baseline—raw execution.

## Requirements
- **Operating System**: 
  - WSL2 (Windows 10/11 with Ubuntu) **OR** Ubuntu Linux (e.g., 20.04/22.04).
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 4060).
- **CUDA Toolkit**: Version 12.4 (or compatible—check NVIDIA driver support).
- **NVIDIA Driver**: Version 550.144.03 (or compatible with CUDA 12.4).
- **Compiler**: `nvcc` (included with CUDA Toolkit).
- **Memory**: ~3 GB free RAM (~2.48 GB used by code).
- **Recommended Editor**: Visual Studio Code (VS Code)—better for CUDA development with extensions.

## Installation and Setup

### For WSL (Windows Subsystem for Linux)
1. **Install WSL2 and Ubuntu**:
   - Open PowerShell (Admin) and run:
wsl --install


- Install Ubuntu (e.g., 20.04) from Microsoft Store—launch it:
wsl -d Ubuntu-20.04



2. **Install NVIDIA WSL Driver**:
- Download from NVIDIA: [WSL CUDA Driver](https://developer.nvidia.com/cuda/wsl/download) (e.g., 550.144.03).
- Install on Windows—reboot if prompted.

3. **Install CUDA Toolkit in WSL**:
- Update Ubuntu:
sudo apt update && sudo apt upgrade -y


- Install CUDA 12.4 (adjust version as needed):
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-12-4-local/7fa2af80.pub
sudo apt update
sudo apt install cuda -y



4. **Install VS Code (Optional but Recommended)**:
- On Windows: Download from [VS Code](https://code.visualstudio.com/)—install.
- Install WSL extension: Open VS Code—Extensions (`Ctrl+Shift+X`)—search "WSL"—install.
- Connect to WSL: `Ctrl+Shift+P`—type "WSL: Connect to WSL"—select Ubuntu—open folder with code.

5. **Clone Repository**:
- In WSL terminal:
git clone <your-repo-url>
cd <repo-name>



6. **Compile and Run**:
- Ensure ~3 GB RAM free:
free -h


- Compile and execute:
nvcc -o sha3 sha3.cu -arch=sm_89 && ./sha3


- Expected output: ~60-70 Gb/s—matches prior WSL runs.

### For Ubuntu Linux
1. **Install NVIDIA Driver**:
- Update system:
sudo apt update && sudo apt upgrade -y


- Install driver (e.g., 550.144.03):
sudo apt install nvidia-driver-550 -y


- Reboot:
sudo reboot


2. **Install CUDA Toolkit**:
- Add CUDA repository:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2204-12-4-local/7fa2af80.pub
sudo apt update
sudo apt install cuda -y



3. **Install VS Code (Optional but Recommended)**:
- Install:
sudo snap install --classic code


- Open VS Code:
code .

- Install CUDA extension: Extensions (`Ctrl+Shift+X`)—search "CUDA"—install.

4. **Clone Repository**:
- In terminal:
git clone <your-repo-url>
cd <repo-name>



5. **Compile and Run**:
- Ensure ~3 GB RAM free:
free -h


- Set persistence mode (optional—improves GPU performance):
sudo nvidia-smi -pm 1


- Compile and execute:
nvcc -o sha3 sha3.cu -arch=sm_89 && ./sha3


- Expected output: ~50-55 Gb/s—Ubuntu uses `malloc`—H2D slower (~62 Gb/s).

## Notes
- **VS Code**: Recommended—syntax highlighting, debugging—open folder with `code .`—better than nano/vim for CUDA.
- **WSL vs. Ubuntu**: WSL uses pinned memory (`cudaMallocHost`)—faster H2D (~65-80 Gb/s)—Ubuntu may need `malloc`—slower (~62 Gb/s)—code auto-adjusts.
- **Output**: Shows processed messages, throughput (Gb/s), first hash—e.g., "Processed 1048576 messages (2304 bytes each) in 0.294 seconds, Throughput: 65.83 Gb/s".

## Troubleshooting
- **Compilation Error**: Check CUDA version—`nvcc --version`—ensure driver matches—`nvidia-smi`.
- **Memory Error**: Free RAM—close apps—check `free -h`.
- **No Output**: Verify GPU—`nvidia-smi`—re-run commands.

## Next Steps
- Add streams (paper’s 271.82 Gb/s)—overlap H2D—push past 88.51 Gb/s—future enhancement.

---