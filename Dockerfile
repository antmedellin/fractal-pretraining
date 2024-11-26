# build command  
# docker build -t fractals .  

FROM nvidia/cuda:12.5.0-devel-ubuntu22.04
# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt upgrade -y

RUN apt install -y \
    git \
    build-essential \
    wget \
    unzip \
    pkg-config \
    cmake \
    pip \
    sudo \
    g++ \
    ca-certificates \
    htop \
    nano \
    libgl1-mesa-glx \
    gdal-bin \    
    python3-tk


RUN pip install \
    torch \
    torchvision\
    omegaconf \
    torchmetrics \
    fvcore \
    iopath \
    xformers\
    submitit \
    matplotlib \
    ipykernel \
    opencv-python \
    scikit-learn \
    albumentations \
    transformers \
    evaluate \
    segmentation-models-pytorch \
    numba \
    tifffile \
    lightning \
    tensorboard \
    torch-tb-profiler \
    pandas \
    matplotlib \
    seaborn 


RUN useradd -m developer 

RUN echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer

USER developer

WORKDIR /home