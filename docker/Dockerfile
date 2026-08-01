FROM ubuntu:22.04

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

LABEL org.opencontainers.image.title="mobile-nano-vlm build environment"
LABEL org.opencontainers.image.description="Public model-conversion dependencies; excludes QAIRT/QNN, Android NDK, models, and project source"
LABEL org.opencontainers.image.licenses="Multiple open-source licenses"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV VIRTUAL_ENV=/opt/mobile-nano-vlm/venv
ENV PATH=/opt/mobile-nano-vlm/venv/bin:${PATH}
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    clang \
    cmake \
    curl \
    file \
    g++ \
    git \
    libc++1 \
    libc++abi1 \
    libgl1 \
    libglib2.0-0 \
    libunwind8 \
    make \
    ninja-build \
    openjdk-17-jdk-headless \
    python3 \
    python3-pip \
    python3-venv \
    tar \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-public.txt /tmp/requirements-public.txt

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
       --index-url "$TORCH_INDEX_URL" \
       torch==2.6.0 \
    && python -m pip install \
       --index-url "$PIP_INDEX_URL" \
       -r /tmp/requirements-public.txt \
    && python -c "import numpy, onnx, torch, transformers, yaml; \
assert numpy.__version__ == '1.26.4'; \
assert onnx.__version__ == '1.19.1'; \
assert torch.__version__.startswith('2.6.0'); \
assert transformers.__version__ == '4.57.6'; \
print('Public Python environment verified')"

RUN printf '%s\n' \
      "This public image contains only open-source build and Python dependencies." \
      "It intentionally excludes:" \
      "- Qualcomm QAIRT/QNN SDK" \
      "- Android SDK/NDK" \
      "- Model weights and generated contexts" \
      "- Project source code" \
      > /opt/mobile-nano-vlm/PUBLIC_IMAGE_CONTENTS.txt \
    && touch /opt/mobile-nano-vlm/public-deps.ready \
    && rm -f /tmp/requirements-public.txt

# QAIRT 2.48 的转换器扩展在运行时会加载 libpython3.10.so.1.0。
# 这里只安装 Ubuntu 的公开开发运行库，不包含任何 Qualcomm 文件。
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# MNN LLM exporter uses yaspin for command-line progress display.
RUN python -m pip install \
      --index-url "$PIP_INDEX_URL" \
      yaspin==3.1.0 \
    && python -c "import yaspin; print('MNN LLM exporter dependencies verified')"

WORKDIR /workspace
CMD ["/bin/bash"]
