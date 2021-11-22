FROM continuumio/miniconda3

LABEL Description="CLOE Docker Image"
WORKDIR /home
ENV SHELL /bin/bash

RUN apt-get update --allow-releaseinfo-change && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main universe restricted multiverse" && \
    apt-get install gfortran -y && \
    apt-get install texlive dvipng texlive-latex-extra texlive-fonts-recommended cm-super -y && \
    apt-get clean

COPY environment.yml .

RUN conda env create -f environment.yml
