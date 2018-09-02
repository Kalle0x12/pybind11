LIS-Solver Python extension
==============================

[![Build Status](https://travis-ci.org/Kalle0x12/Test2.svg?branch=master)](https://travis-ci.org/Kalle0x12/Test2)

Pybind11 https://github.com/pybind/pybind11 is a lightweight header-only library
that exposes C++ types in Python and vice versa, mainly to create Python bindings
of existing C++ Code or as in this case of C Code. A great advantage of pybind11
is that Python NumPy arrays can easily be passed around between C++/C and Python
without making deep copies.

LIS (Library of Iterative Solvers for linear systems) http://www.ssisc.org/lis/
is a parallel software library for solving sparse linear equations and eigenvalue
problems, which uses OpenMP or MPI for parallel computing environments.

This example shows how to call a sparse solver routine implemented with LIS from
Python. A system of sparse linear equations in CSR (Compressed Sparse Row)
representation is delivered from Python to LIS by means of a set of NumPy arrays. 
The solution vector x of the equation A x = b is returned as a NumPy array to Python
in place. The solver's configuration (type of solver or preconditioner and other parameters)
is passed as a command string from Python to LIS.

To be continued...
