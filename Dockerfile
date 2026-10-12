FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y wget ca-certificates git bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh 

WORKDIR /workspace
COPY environment.yml /workspace/environment.yml

ENV PYTHONPATH=/workspace

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda -y && \
    conda env create -f /workspace/environment.yml -n fraud_env || conda env create -f /workspace/environment.yml -n fraud_env

SHELL ["/bin/bash", "-lc"]
RUN echo "conda activate fraud_env" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV=fraud_env

CMD ["/bin/bash"]
