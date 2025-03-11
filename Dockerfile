# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variable to disable GPU use in Hugging Face Transformers
ENV USE_CPU=True

# Command to run the FastAPI app
CMD ["uvicorn", "api_rag:app", "--host", "0.0.0.0", "--port", "8000"]