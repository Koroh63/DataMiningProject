FROM python:3.11

WORKDIR /app

# Install necessary dependencies
COPY requirements.txt ./
RUN pip install streamlit-pandas-profiling --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY /app .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.fileWatcherType","none"]