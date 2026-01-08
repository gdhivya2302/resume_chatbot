FROM python:3.10-slim

WORKDIR /decoder

# Install system dependencies (important for sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app file
COPY c66.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "c66.py", "--server.address=0.0.0.0", "--server.port=8501"]


