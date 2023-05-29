LABEL authors="SYLVESTRE APETCHO"

# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system libraries
RUN apt-get update && apt-get install -y libgl1-mesa-dev

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run Object_detection_App.py when the container launches
CMD ["streamlit", "run", "Object_detection_App.py"]
