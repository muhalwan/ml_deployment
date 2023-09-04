# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /ml_deployment_pipeline

# Copy the current directory contents into the container
COPY . /ml_deployment_pipeline

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set an entry point to execute when the container starts (you can adjust this based on your needs)
CMD ["python", "src/train_model.py"]
