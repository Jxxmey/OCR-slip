# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for Google Cloud Vision (optional, but good for robustness)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    # Add any other system dependencies if needed, though Vision API usually handles most of it.
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Ensure the Google Cloud Service Account key is at the expected path
# Note: In Render, it's better to provide credentials via Environment Variables
# rather than copying the key file directly into the Docker image.
# If you must copy it, make sure it's *not* committed to Git.
# For this example, we're assuming GOOGLE_APPLICATION_CREDENTIALS points to a path
# in Render's environment, or you'll inject it via Render's environment variables.
# For simplicity, if shiba-bot.json is needed, ensure it's here during build,
# but the best practice for Render is setting GOOGLE_APPLICATION_CREDENTIALS
# as a Base64 encoded string or directly as a secret file via Render's dashboard.
# For local testing, your .env works.

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# This needs to be set in Render's environment variables, not hardcoded here.
# Render will automatically map your defined environment variables.
# For local testing within Docker, you could put:
# ENV GOOGLE_APPLICATION_CREDENTIALS="/app/shiba-bot.json"
# But for Render, use their UI to set it for better security.

# Expose the port the app runs on
EXPOSE 8000

# Run the Uvicorn server when the container launches
# Use $PORT provided by Render for the listening port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]