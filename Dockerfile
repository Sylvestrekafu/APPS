FROM ubuntu:latest
LABEL authors="SYLVESTRE APETCHO"

ENTRYPOINT ["top", "-b"]

FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY Object_detection_App.py ./Object_detection_App.py
ENTRYPOINT ["streamlit", "run"]
CMD ["Object_detction_App.py"]