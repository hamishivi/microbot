# FROM tensorflow/tensorflow:latest-py3
FROM tiangolo/uvicorn-gunicorn-starlette:python3.7

COPY main.py /app/main.py
COPY chatbot.py /app/chatbot.py
COPY static /app/static
COPY data /app/data
COPY templates /app/templates
COPY requirements.txt /app/requirements.txt
COPY gunicorn_conf.py /app/gunicorn_conf.py

RUN pip install -r /app/requirements.txt

RUN pip install gunicorn

WORKDIR /app
