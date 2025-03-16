From python:3.7

RUN mkdir/app

COPY ./* /app

RUN pip install -r requirements.txt