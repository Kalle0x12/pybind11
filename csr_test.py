from __future__ import print_function
import lis_wrapper
import numpy as np
import scipy.sparse

# Define a symmetric 8 x 8 dense upper triangular matrix first. 
# This matrix is part of the examples which come with Intel's MKL library
# and is used here for historical reasons.

# A:
#     7.0,       1.0,           2.0, 7.0,
#          -4.0, 8.0,      2.0,
#                1.0,                     5.0,
#                     7.0,           9.0,
#                          5.0, 1.0, 5.0,
#                              -1.0,      5.0,
#                                   11.0,
#                                         5.0

A = np.zeros((8, 8), dtype=np.float64)
A[0, 0] = 7.0
A[0, 2] = 1.0
A[0, 5] = 2.0
A[0, 6] = 7.0

A[1, 1] = -4.0
A[1, 2] = 8.0
A[1, 4] = 2.0

A[2, 2] = 1.0
A[2, 7] = 5.0

A[3, 3] = 7.0
A[3, 6] = 9.0

A[4, 4] = 5.0
A[4, 5] = 1.0
A[4, 6] = 5.0

A[5, 5] = -1.0
A[5, 7] = 5.0

A[6, 6] = 11.0

A[7, 7] = 5.0

# print "Dense matrix:"
print(A)
# Dense matrix to sparse matrix in CSR format
Acsr = scipy.sparse.csr_matrix(A)

print("Sparse upper triangular CSR matrix:")
print("values:  ", Acsr.data)
# Indices are 0 based
print("index:   ", Acsr.indices)
print("pointer: ", Acsr.indptr)

# LIS Manual: Appendix File Formats
# "Note that both the upper and lower triangular entries need to be stored
# irrespective of whether the matrix is symmetric or not."

# Convert the upper triangular CSR matrix Acsr to 'full' CSR matrix Acsr_full
Acsr_full = Acsr + Acsr.T - scipy.sparse.diags(Acsr.diagonal())

print()
print("Sparse 'full' CSR matrix:")
print("values:  ", Acsr_full.data)
# Indices are 0 based
print("index:   ", Acsr_full.indices)
print("pointer: ", Acsr_full.indptr)

# initial guess for solution x
x = np.zeros(8)
# right hand side
b = np.ones(8)
info = 1    # make LIS more verbose
tol = 1e-6  # convergence tolerance
max_iter = 10000 # maximum number of iterations
logfname = "residuals.log" # log

# in lis_cmd following parameters are set:
# -i cg : conjugate gradient solver
# -p ssor : SSOR preconditioner
# -tol  : convergence tolerance
# -maxiter : maximum number of iterations
# -p ssor : SSOR preconditioner
# -ssor_w 1.0 : relaxation coefficient w (0 < w < 2)
# -initx_zeros 0 : don't set initial values for x to 0. The initial guess is passed by x to LIS
# -print mem : Save the residual history to logfile

lis_cmd = "-i cg -tol %e -maxiter %d -p ssor -ssor_w 1.0 -initx_zeros 0 -print mem" % (tol, max_iter)
lis_wrapper.lis(Acsr_full.data, Acsr_full.indices, Acsr_full.indptr, x, b, info, lis_cmd, logfname)

# check solution x with original dense matrix A first
# convert upper triangular matrix AA to 'full' matrix
y = (A + A.T - np.eye(A.shape[0]) * A.diagonal()).dot(x)
assert (np.allclose(b, y))

# check solution  with sparse matrix Acsr_full
y = Acsr_full.dot(x)
assert (np.allclose(b, y))
print("Solution x: ", x)
print()
print("A * x:", y)
print("b    :", b)
