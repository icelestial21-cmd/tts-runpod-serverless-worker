FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Build args
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV WORKER_MODEL_DIR=/app/model
ENV WORKER_USE_CUDA=True

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV WORKER_DIR=/app
RUN mkdir ${WORKER_DIR}
WORKDIR ${WORKER_DIR}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
                && chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && \
    rm ${WORKER_DIR}/requirements.txt
RUN python3 -c "import deepspeed; print(deepspeed.__version__)"

# Install Python dependencies (Worker Template)
COPY builder/requirements_audio_enhancer.txt ${WORKER_DIR}/requirements_audio_enhancer.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements_audio_enhancer.txt && \
    rm ${WORKER_DIR}/requirements_audio_enhancer.txt

# Download models at build-time to avoid cold-start timeouts
RUN pip install huggingface-hub
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='coqui/XTTS-v2', local_dir='${WORKER_MODEL_DIR}/xttsv2')"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ResembleAI/resemble-enhance', local_dir='${WORKER_MODEL_DIR}/audio_enhancer')"

# Add src files (Worker Template)
ADD src ${WORKER_DIR}

ENV RUNPOD_DEBUG_LEVEL=INFO

CMD python3 -u ${WORKER_DIR}/rp_handler.py --model-dir="${WORKER_MODEL_DIR}"
