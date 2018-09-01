#!/bin/bash
set -ev
wget http://www.paralution.com/downloads/paralution-1.1.0.tar.gz
tar xzf paralution-1.1.0.tar.gz
# Bugfix for paralution
patch -p0 < paralution.patch
# Switch OpenCL off for mac os build
patch -p0 < paralution-cmake.patch
# activate -fpermissive for gcc7
if [ $1 = "7" ]; then
   patch -p0 < paralution-cmake-2.patch
fi
cd paralution-1.1.0
mkdir build
cd build
# Dont't build examples
cmake -DBUILD_EXAMPLES=OFF ..
make -j2
