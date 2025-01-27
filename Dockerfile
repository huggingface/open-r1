FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -e '.[dev]'
COPY . .
