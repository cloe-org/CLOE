FROM ubuntu:bionic

LABEL Description="CLOE Docker Image"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system dependencies + distutils and build tools
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      git \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      make \
      mercurial \
      openssh-client \
      procps \
      subversion \
      wget \
      python3-distutils \
      build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Arguments for caching Miniconda installer
# renovate: datasource=custom.miniconda_installer
ARG INSTALLER_URL_LINUX64="https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-x86_64.sh"
ARG SHA256SUM_LINUX64="96a44849ff17e960eeb8877ecd9055246381c4d4f2d031263b63fa7e2e930af1"
ARG INSTALLER_URL_S390X="https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-s390x.sh"
ARG SHA256SUM_S390X="b601cb8e3ea65a4ed1aecd96d4f3d14aca5b590b2e1ab0ec5c04c825f5c5e439"
ARG INSTALLER_URL_AARCH64="https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-aarch64.sh"
ARG SHA256SUM_AARCH64="05f70cbc89b6caf84e22db836f7696a16b617992eb23d6102acf7651eb132365"

# Install Miniconda
RUN set -eux && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        INSTALLER_URL="${INSTALLER_URL_LINUX64}"; \
        SHA256SUM="${SHA256SUM_LINUX64}"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        INSTALLER_URL="${INSTALLER_URL_S390X}"; \
        SHA256SUM="${SHA256SUM_S390X}"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        INSTALLER_URL="${INSTALLER_URL_AARCH64}"; \
        SHA256SUM="${SHA256SUM_AARCH64}"; \
    fi && \
    wget "${INSTALLER_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    sha256sum --check --status shasum && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Copy and create the Conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml -n cloe && \
    conda clean -afy
