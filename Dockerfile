# Use Python 3.11-alpine as the base image for a lightweight build
FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install required system dependencies
RUN apk add --no-cache \
    build-base \
    musl-dev \
    libffi-dev \
    python3-dev \
    ffmpeg \
    gfortran \
    && rm -rf /var/cache/apk/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
