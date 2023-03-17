FROM python:3.8
FROM nvidia/cuda:12.1.0-devel-ubuntu18.04
FROM scipy/scipy-dev:latest
FROM pytorch/pytorch

RUN apt-get update
RUN apt install build-essential -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .
RUN python -m pip install --no-cache -r requirements.txt

COPY . /vgl-ozu
WORKDIR /vgl-ozu

