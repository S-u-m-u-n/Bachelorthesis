#include <stdlib.h>
#include "matmul.h"

int main(int argc, char** argv) {
  double * __restrict__ A = (double*) calloc(4000000, sizeof(double));
  double * __restrict__ B = (double*) calloc(4000000, sizeof(double));
  double * __restrict__ __return = (double*) calloc(4000000, sizeof(double));

  __dace_init_matmul(A, B, __return);
  __program_matmul(A, B, __return);
  __dace_exit_matmul(A, B, __return);

  free(A);
  free(B);
  free(__return);
  return 0;
}
