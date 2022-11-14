FROM continuumio/miniconda3

LABEL Description="CLOE Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

ARG CC=gcc-9
ARG CXX=g++-9

RUN apt-get update --allow-releaseinfo-change && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main universe restricted multiverse" && \
    apt-get install gfortran -y && \
    apt-get install gcc-9 g++-9 -y && \
    apt-get install texlive dvipng texlive-latex-extra texlive-fonts-recommended cm-super -y && \
    apt-get clean

RUN ln -s /usr/bin/g++-9 /usr/bin/g++

COPY environment.yml .

RUN conda env create -f environment.yml
