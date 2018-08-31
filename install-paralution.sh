#!/bin/bash
set -ev
wget http://www.paralution.com/downloads/paralution-1.1.0.tar.gz
tar xzf paralution-1.1.0.tar.gz
# Bugfix for paralution
patch -p0 < paralution.patch
# Switch OpenCL off for mac os build
patch -p0 < paralution-cmake.patch
cd paralution-1.1.0
mkdir build
cd build
# -DCMAKE_CXX_FLAGS="-fpermissive"
# Dont't build examples

if [ $1 = "7" ]; then
    cmake -DBUILD_EXAMPLES=OFF -DCMAKE_CXX_FLAGS="-fpermissive" ..
else
    cmake -DBUILD_EXAMPLES=OFF ..
fi
make -j2
