#!/usr/bin/env bash
set -euo pipefail

# Build OpenPose with Python bindings on Ubuntu.
# The script installs required dependencies, ensures protobuf headers are
# available and builds OpenPose from source with matching protobuf versions.


sudo apt-get update

# Discover the candidate protobuf version available from APT. Some Ubuntu
# images may not offer libprotobuf-dev directly, returning an empty or '(none)'
# candidate. In that case fall back to libprotoc-dev and determine its version
# after installation.
proto_pkg_ver=$(apt-cache policy libprotobuf-dev | awk '/Candidate:/ {print $2}')

if [ -z "${proto_pkg_ver}" ] || [ "${proto_pkg_ver}" = "(none)" ]; then
    echo "libprotobuf-dev not available; using libprotoc-dev instead"
    sudo apt-get install -y build-essential cmake git libopencv-dev \
        libgoogle-glog-dev libatlas-base-dev libprotoc-dev
    proto_pkg_ver=$(dpkg -s libprotoc-dev | awk '/^Version:/ {print $2}')
else
    sudo apt-get install -y build-essential cmake git libopencv-dev \
        libgoogle-glog-dev libatlas-base-dev \
        "libprotobuf-dev=${proto_pkg_ver}" "protobuf-compiler=${proto_pkg_ver}"
fi

# Extract upstream version (strip epoch and revision) for source builds.
proto_src_ver=$(echo "$proto_pkg_ver" | cut -d'-' -f1 | cut -d':' -f2)

# Some Ubuntu images may not ship the protobuf headers even after installing
# libprotobuf-dev. OpenPose requires google/protobuf/runtime_version.h.
if ! grep -q "runtime_version.h" -r /usr/include/google/protobuf 2>/dev/null; then
    echo "protobuf headers missing; building protobuf from source"
    wget -q "https://github.com/protocolbuffers/protobuf/releases/download/v${proto_src_ver}/protobuf-cpp-${proto_src_ver}.tar.gz"
    tar -xzf protobuf-cpp-${proto_src_ver}.tar.gz
    cd protobuf-${proto_src_ver}
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

