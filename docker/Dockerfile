FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3 python3-pip \
    clang llvm \
    cmake make \
    curl wget unzip \
    git vim \
    openjdk-17-jdk \
    libc++1 libc++abi1 libunwind8 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace
CMD ["/bin/bash"]
