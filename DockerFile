FROM python:3.9-slim

WORKDIR /app

# Install necessary dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080"]