FROM python:3.12.12-slim-trixie

WORKDIR /app

COPY requirements.txt /app
COPY app /app

RUN pip install --no-cache-dir -r requirements.txt

#ENTRYPOINT jupyter notebook --allow-root --ip 0.0.0.0 --no-browser
