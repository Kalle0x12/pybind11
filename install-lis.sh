#!/bin/bash
set -ev
git clone https://github.com/anishida/lis.git
patch -p0 < lis.patch
# make ckeck fails on osx
cd lis && ./configure --prefix=$HOME/lis --enable-omp  --enable-shared  && make -j2  && make install
