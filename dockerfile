# Use the latest official Python image
FROM python:3.12.0-slim

# Set environment variables to avoid .pyc files and enable unbuffered output
# ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install dependencies
# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy any necessary scripts (optional) – these will be overridden by the bind mount,
# but they are useful for initial container builds if no volume is provided.
# COPY pylint.sh .

# Make sure your shell script has execution permissions
# RUN chmod +x pylint.sh

# Set the default command to bash
CMD ["bash"]
