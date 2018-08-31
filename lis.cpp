#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <stdio.h>
#include <string>

#include "lis_config.h"
#include "lis.h"
//#pragma GCC diagnostic ignored "-Wwrite-strings"

namespace py = pybind11;
using namespace std;

void wrapper(py::array_t<double> values, py::array_t<int> columns,
        py::array_t<int> index, py::array_t<double> x,
        py::array_t<double> b, int info, std::string lis_cmd, std::string fname) {

    LIS_MATRIX A;
    LIS_VECTOR X, B;
    LIS_SOLVER solver;
    char **argv = NULL;
    LIS_INT err, argc = 0;
    LIS_INT iter, iter_double, iter_quad, nsol, nprecon;
    double times, itimes, ptimes, p_c_times, p_i_times;
    LIS_REAL resid;
    char solvername[128], preconname[128];
    LIS_INT len_x;
    
    py::buffer_info info_values = values.request();
    auto ptr_values = static_cast<LIS_SCALAR *> (info_values.ptr);
    py::buffer_info info_columns = columns.request();
    auto ptr_columns = static_cast<LIS_INT *> (info_columns.ptr);
    py::buffer_info info_index = index.request();
    auto ptr_index = static_cast<LIS_INT *> (info_index.ptr);
    py::buffer_info info_x = x.request();
    auto ptr_x = static_cast<LIS_SCALAR *> (info_x.ptr);
    py::buffer_info info_b = b.request();
    auto ptr_b = static_cast<LIS_SCALAR *> (info_b.ptr);
    // convert std::string to char*
    char *cmd = new char[lis_cmd.length() + 1];
    strcpy(cmd, lis_cmd.c_str());
    char *logf = new char[fname.length() + 1];
    strcpy(logf, fname.c_str());
    // number of equations
    len_x = (LIS_INT)info_x.shape[0];
    
    printf("LIS start...\n");
    LIS_DEBUG_FUNC_IN;
    err = lis_initialize(&argc, &argv);
    CHKERR(err);
    //create and associate the coefficient matrix in CSR format
    err = lis_matrix_create(0, &A);
    CHKERR(err);
    err = lis_matrix_set_size(A, 0, len_x);
    CHKERR(err);
    err = lis_matrix_set_csr((LIS_INT)info_values.shape[0], ptr_index, ptr_columns, ptr_values, A);
    CHKERR(err);
    err = lis_matrix_assemble(A);
    CHKERR(err);
    // create rhs and solution vectors
    err = lis_vector_create(0, &B);
    CHKERR(err);
    err = lis_vector_create(0, &X);
    CHKERR(err);
    err = lis_vector_set_size(B, 0, len_x);
    CHKERR(err);
    err = lis_vector_set_size(X, 0, len_x);
    CHKERR(err);
    printf("\nLIS: OpenMP Infos... \n");
#ifdef _OPENMP
#ifdef _LONG__LONG
    printf("LIS: max number of threads = %lld\n", omp_get_num_procs());
    printf("LIS: number of threads = %lld\n", omp_get_max_threads());
#else
    printf("LIS: max number of threads = %d\n", omp_get_num_procs());
    printf("LIS: number of threads = %d\n", omp_get_max_threads());
#endif
#endif    
    // set solution vector X to 0
    //err = lis_vector_set_all(0.0, X);
    //CHKERR(err);
    // setup X
    for (LIS_INT i = 0; i < len_x; i++) {
        lis_vector_set_value(LIS_INS_VALUE, i, *(ptr_x + i), X);
    }
    // setup rhs
    for (LIS_INT i = 0; i < len_x; i++) {
        lis_vector_set_value(LIS_INS_VALUE, i, *(ptr_b + i), B);
    }
    // create solver
    err = lis_solver_create(&solver);
    CHKERR(err);
    //pass command string to LIS
    err = lis_solver_set_option(cmd, solver);
    CHKERR(err);
    err = lis_solve(A, B, X, solver);
    CHKERR(err);
    lis_solver_get_iterex(solver, &iter, &iter_double, &iter_quad);
    lis_solver_get_residualnorm(solver, &resid);
    lis_solver_get_solver(solver, &nsol);
    lis_solver_get_solvername(nsol, solvername);
    lis_solver_get_precon(solver, &nprecon);
    lis_solver_get_preconname(nprecon, preconname);
    printf("\nLIS: statistical infos...\n");
#ifdef _LONGLONG
    printf
            ("%s: number of iterations     = %lld (double = %lld, quad = %lld)\n",
            solvername, iter, iter_double, iter_quad);
#else
    printf("%s: number of iterations     = %d (double = %d, quad = %d)\n",
            solvername, iter, iter_double, iter_quad);
#endif
    if (info) {
        printf("LIS command string: %s\n", cmd);
        printf("Logfile: %s\n", logf);
        lis_solver_get_timeex(solver, &times, &itimes, &ptimes, &p_c_times, &p_i_times);
        printf("%s: elapsed time             = %e sec.\n", solvername, times);
        printf("%s:   preconditioner         = %e sec.\n", preconname, ptimes);
        printf("%s:   matrix creation        = %e sec.\n", solvername, p_c_times);
        printf("%s:   linear solver          = %e sec.\n", solvername, itimes);
    }
#ifdef _LONG_DOUBLE
    printf("%s: relative residual 2-norm = %Le\n\n", solvername, resid);
#else
    printf("%s: relative residual 2-norm = %e\n\n", solvername, resid);
#endif
    //copy solution vector back to Python
    for (LIS_INT i = 0; i < len_x; i++) {
        lis_vector_get_value(X, i, (ptr_x + i));
    }
    // write residuals to logfile
    lis_solver_output_rhistory(solver, logf);
    // clean up
    lis_vector_destroy(X);
    lis_vector_destroy(B);
    lis_matrix_unset(A);
    lis_matrix_destroy(A);
    lis_solver_destroy(solver);
    lis_finalize();
    delete[] cmd;
    delete[] logf;
    LIS_DEBUG_FUNC_OUT;
    printf("LIS terminated...\n");
}

PYBIND11_MODULE(lis_wrapper, m) {
    m.doc() = "Test LIS interface";
    m.def("lis", &wrapper, "Call LIS wrapper");
}

