FROM gitlab.euclid-sgs.uk:4567/pf-ist-likelihood/likelihood-implementation

LABEL Description="CLOE Docker Image"
WORKDIR /home
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash

COPY . /home

RUN conda activate cloe && \
    python -m pip install .

RUN echo 'export PATH="/opt/conda/envs/cloe/bin:$PATH"' >> ~/.bashrc

RUN echo -e '#!/bin/bash\njupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root' > /usr/bin/notebook && chmod +x /usr/bin/notebook
