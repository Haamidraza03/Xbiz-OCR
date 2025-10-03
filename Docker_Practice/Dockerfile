# Use a common, stable Python base image
FROM python:3.11-slim


# Set the working directory to /app
WORKDIR /app

# Copy the requirements file and install dependencies first (for efficient layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# The EXPOSE instruction should match the port your Flask app runs on (3000)
EXPOSE 3000

# Command to run your application
CMD ["python", "app.py"]