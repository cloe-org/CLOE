FROM continuumio/miniconda3

LABEL Description="CLOE Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

RUN apt-get update
RUN apt-get install build-essential -y && \
    apt-get install gfortran -y

COPY environment.yml .

RUN conda env create -f environment.yml
