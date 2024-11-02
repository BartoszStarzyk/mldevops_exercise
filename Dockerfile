FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS base

# Install project dependencies
RUN pip install matplotlib
# -------------------------------------------------------------------------------------------- #

# OpenCV & mysqlclient dependencies
RUN apt update && apt install -y \
        build-essential \
        default-libmysqlclient-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        pkg-config \
        python3-dev \
        vim \
    && apt clean

# Create a non-root user
ENV UID 1000
ENV GID 1000
RUN addgroup --gid ${GID} --system user && \
    adduser --shell /bin/bash --disabled-password --uid ${UID} --system --group user

CMD ["bash"]
