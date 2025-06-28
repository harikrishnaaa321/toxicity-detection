# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run your FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -t toxicity-detector-app .
# docker run -d -p 8000:8000 toxicity-detector-app
