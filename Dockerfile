FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

#CMD ["python", "workloads.py"]
CMD ["prefect", "server", "start"]
