FROM python:3.11

WORKDIR /app

# Install necessary dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir visions==0.7.5
RUN pip install --no-cache-dir joblib==1.4.2 --no-cache-dir
RUN pip install streamlit-pandas-profiling

# Copy the rest of the application code
COPY /app .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.fileWatcherType","none"]