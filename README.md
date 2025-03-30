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
## ⚡ Features
✅ Fully isolated deep learning environment  
✅ Persistent storage of all changes  
✅ Jupyter Notebook and Python script support  
✅ GPU acceleration (if supported and configured)  
✅ Easy setup with `docker-compose`

---

This setup ensures a **consistent, reproducible, and efficient** workflow for deep learning projects. Happy coding! 🚀
