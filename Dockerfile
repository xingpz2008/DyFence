FROM --platform=linux/amd64 ubuntu:22.04@sha256:0d779ea97881505f5ef0039336ee85edba27519bdba968c284c86ee066a973c8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        iproute2 \
        libjemalloc-dev \
        libomp-dev \
        libssl-dev \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

COPY frontend/requirements.txt frontend/requirements.txt
RUN python3 -m pip install --no-cache-dir pip==25.0.1 \
    && python3 -m pip install --no-cache-dir -r frontend/requirements.txt

COPY . .

# A single build job keeps the image build within the baseline resource budget.
RUN python3 ae.py build --build-dir build-ae --jobs 1

ENTRYPOINT ["python3", "ae.py"]
CMD ["--help"]
