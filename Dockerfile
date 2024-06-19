FROM python:3.11

WORKDIR /app

# Install necessary dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall joblib visions -y
RUN pip install visions==0.7.5 --no-cache-dir
RUN pip install joblib==1.4.2 --no-cache-dir
RUN pip install streamlit-pandas-profiling --no-cache-dir

# Copy the rest of the application code
COPY /app .

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.fileWatcherType","none"]