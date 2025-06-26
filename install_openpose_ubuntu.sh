#!/usr/bin/env bash
set -euo pipefail

# Build OpenPose with Python bindings on Ubuntu.
# The script installs required dependencies, ensures protobuf headers are
# available and builds OpenPose from source.

sudo apt-get update
sudo apt-get install -y build-essential cmake git libopencv-dev \
    libgoogle-glog-dev libatlas-base-dev libprotobuf-dev protobuf-compiler

# Some Ubuntu images may not ship the protobuf headers even after installing
# libprotobuf-dev. OpenPose requires google/protobuf/runtime_version.h.
if ! grep -q "runtime_version.h" -r /usr/include/google/protobuf 2>/dev/null; then
    echo "protobuf headers missing; building protobuf from source"
    # Match the version to the installed runtime library.
    proto_ver=$(protoc --version | awk '{print $2}')
    wget -q https://github.com/protocolbuffers/protobuf/releases/download/v${proto_ver}/protobuf-cpp-${proto_ver}.tar.gz
    tar -xzf protobuf-cpp-${proto_ver}.tar.gz
    cd protobuf-${proto_ver}
    ./configure
    make -j"$(nproc)"
    sudo make install
    sudo ldconfig
    cd ..
fi

# Clone and build OpenPose
if [ ! -d openpose ]; then
    git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
fi
cd openpose
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=OFF ..
make -j"$(nproc)"
# The Python module will be available under build/python.

