#include <stdlib.h>
#include "getStarted.h"

int main(int argc, char** argv) {
  double * __restrict__ A = (double*) calloc(6, sizeof(double));
  double * __restrict__ __return = (double*) calloc(6, sizeof(double));

  __dace_init_getStarted(A, __return);
  __program_getStarted(A, __return);
  __dace_exit_getStarted(A, __return);

  free(A);
  free(__return);
  return 0;
}
