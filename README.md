# Finder Ai Repository

This repository provides a **Docker-based development environment** for deep learning projects, ensuring consistency and reproducibility across different setups. The environment supports Python scripts, Jupyter notebooks, and deep learning model training inside a container while preserving all changes made to files.

## 📁 Project Structure
```
repo-root/
│── data/         # Dataset storage
│── models/       # Trained models and checkpoints
│── notebooks/    # Jupyter notebooks for experiments
│── reports/      # Logs, metrics, and training results
│── src/          # Source code for model training and utilities
│── .gitignore    # Git ignore file
│── docker-compose.yaml  # Configuration for running the container
│── Dockerfile    # Docker image setup
│── pylint.sh     # Script for linting code
│── README.md     # Documentation
│── requirements.txt  # Python dependencies
│── .devcontainer/  # VS Code Dev Container configuration (optional)
```

## 🐳 Environment Setup with Docker
This project uses **Docker and Docker Compose** to set up the development environment.

### 🔧 **Requirements**
Ensure you have the following installed:
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 🚀 **Setup and Run the Container**
To build and start the container, run:
```bash
docker-compose up -d --build
```
This will:
✅ Build the Docker image using the `Dockerfile`
✅ Mount the entire repository into the container (ensuring file persistence)
✅ Install all dependencies from `requirements.txt`
✅ Provide an isolated development environment with Python and Jupyter support

### 🔄 **Stopping the Container**
To stop and remove the running container:
```bash
docker-compose down
```

## 🖥️ Working Inside the Container
Once the container is running, you can:

### 🏗️ **Run Python Scripts**
```bash
docker exec -it my_deep_learning_container python src/train.py
```

## 🖥️ Optional: Using VS Code Dev Containers
If you use VS Code, you can take advantage of the **Dev Containers** extension to work inside the container seamlessly.

### 📦 **Opening in Dev Containers**
1. Install the [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open VS Code in the repository root.
3. When prompted, select **Reopen in Container**. Alternatively, open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on Mac) and select:
   ```
   Remote-Containers: Reopen in Container
   ```
4. VS Code will build and start the container, automatically mounting the repository.

### ⚙️ **Configuration**
The `.devcontainer/devcontainer.json` file configures the environment inside the container, including extensions and settings:
```json
{
  "name": "FinderAi Dev Container",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/app",
  "shutdownAction": "stopCompose",
  "customizations": {
    "vscode": {
      "settings": {
        "terminal.integrated.shell.linux": "/bin/bash"
      },
      "extensions": [
        "ms-python.python",
        "ms-azuretools.vscode-docker"
      ]
    }
  }
}
```
This ensures a fully configured environment whenever you open the repository with Dev Containers.

## ⚡ Features
✅ Fully isolated deep learning environment  
✅ Persistent storage of all changes  
✅ Jupyter Notebook and Python script support  
✅ GPU acceleration (if supported and configured)  
✅ Easy setup with `docker-compose`  
✅ Seamless integration with VS Code Dev Containers (optional)  

---

This setup ensures a **consistent, reproducible, and efficient** workflow for deep learning projects. Happy coding! 🚀

### To save changes from inside and outside the container
```bash
sudo chmod -R a+rwX .
sudo chown -R $USER:$USER .
```