FROM python:3-slim

# Copy only requirements.txt for better layer caching
COPY requirements.txt /app/

# Install deps
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy code and model
COPY . /app
WORKDIR /app

# Port and CMD
EXPOSE 8000
CMD ["python3", "api_server.py"]
