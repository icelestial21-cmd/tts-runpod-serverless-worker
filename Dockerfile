# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV WORKER_TTS_MODEL_DIR=/app/model/tts
ENV WORKER_AUDIO_ENHANCER_DIR=/app/model/audio_enhancer
ENV WORKER_USE_CUDA=True
ENV WORKER_DIR=/app

# Use bash with pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create application directory and set as working directory
RUN mkdir ${WORKER_DIR}
WORKDIR ${WORKER_DIR}

# Set additional environment variables for non-interactive apt
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and set proper permissions
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Set HOME and SHELL for non-root user
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Copy requirements file and install Python dependencies
COPY requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip3 install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && rm ${WORKER_DIR}/requirements.txt

# --- Download models during build so that they are baked into the image ---
# Download the RUSynth model from Hugging Face (it saves config.json, model.onnx, dictionary.txt)
RUN python3 -c "from rusynth import RUSynth; RUSynth.from_pretrained(repo_id='bes-dev/rusynth', local_dir='${WORKER_TTS_MODEL_DIR}')"

# Download the Audio Enhancer model (this command downloads necessary files into enhancer_stage2 folder)
RUN python3 -c "from resemble_enhance.audio_enhancer import AudioEnhancer; AudioEnhancer.setup(install_dir='${WORKER_AUDIO_ENHANCER_DIR}')"
# --- End download models ---

# Copy source code (handler and schema)
COPY rp_handler.py ${WORKER_DIR}/rp_handler.py
COPY rp_schema.py ${WORKER_DIR}/rp_schema.py

# Set debug level if needed
ENV RUNPOD_DEBUG_LEVEL=INFO

# Start the worker
CMD ["python3", "-u", "${WORKER_DIR}/rp_handler.py"]
