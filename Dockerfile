FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Create a conda environment
RUN conda create -n myenv python=3.10

# Activate the environment
SHELL ["/bin/bash", "-c"]
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install pip packages
# We'll split this into multiple RUN commands to avoid hitting limits
# First round of installations (base packages)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    matplotlib==3.9.4 \
    scipy==1.13.1 \
    scikit-learn==1.6.0 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    transformers==4.46.3 \
    huggingface-hub==0.28.1

# Second round
RUN pip install --no-cache-dir \
    ipython==8.21.0 \
    jupyter==1.0.0 \
    jupyterlab==3.6.8 \
    pillow==11.0.0 \
    plotly==5.24.1 \
    seaborn==0.13.2 \
    tqdm==4.67.1

# Third round - Google and cloud packages
RUN pip install --no-cache-dir \
    google-api-python-client==1.8.0 \
    google-auth==2.37.0 \
    google-cloud-aiplatform==1.75.0 \
    google-cloud-storage==2.14.0 \
    google-cloud-bigquery==3.25.0 \
    fsspec==2024.12.0 \
    gcsfs==2024.12.0

# Fourth round - Other packages
RUN pip install --no-cache-dir \
    accelerate==1.4.0 \
    aiohttp==3.11.11 \
    beautifulsoup4 \
    fastapi==0.115.6 \
    requests \
    pydantic==2.10.4 \
    uvicorn==0.34.0 \
    ray==2.40.0

# Fifth round - Remaining packages
RUN pip install --no-cache-dir \
    absl-py==2.1.0 \
    aiohappyeyeballs==2.4.4 \
    aiofiles==22.1.0 \
    biopython==1.85 \
    cryptography==44.0.0 \
    einops==0.8.1 \
    networkx==3.4.2 \
    peft==0.14.0 \
    safetensors==0.5.2 \
    tokenizers==0.20.3

# Create a requirements.txt file for remaining packages
COPY paste.txt /tmp/requirements.txt

# Attempt to install remaining packages (some may fail but we continue)
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true

# Set the working directory
WORKDIR /workspace

# Command to run when the container starts
CMD ["bash"]
