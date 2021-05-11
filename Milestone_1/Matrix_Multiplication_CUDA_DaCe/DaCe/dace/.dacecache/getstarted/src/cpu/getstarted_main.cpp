#include <stdlib.h>
#include "getstarted.h"

int main(int argc, char** argv) {
  double * __restrict__ A = (double*) calloc(6, sizeof(double));
  double * __restrict__ __return = (double*) calloc(6, sizeof(double));

  __dace_init_getstarted(A, __return);
  __program_getstarted(A, __return);
  __dace_exit_getstarted(A, __return);

  free(A);
  free(__return);
  return 0;
}
