#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lis/lib
echo $LD_LIBRARY_PATH
$1 csr_test.py
