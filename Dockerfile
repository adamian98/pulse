# Use nvidia docker image as base image to have GPU support
FROM nvidia/cuda:10.2-runtime
# /!\ the cuda version depends on your computer. Adapt the above line correspondingly.

ARG NAME=pulse
# Install ubuntu libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-dev \
    libz-dev libopenblas-dev libatlas-base-dev libgtk-3-dev \
    libboost-filesystem-dev build-essential cmake pkg-config curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV PATH /opt/conda/bin:$PATH
RUN curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/envs/gpuenvs/bin:$PATH

WORKDIR /home/${NAME}

# Install the python dependencies
COPY . .
RUN conda update -n base -c defaults conda
RUN conda env create -f pulse.yml

CMD /bin/bash