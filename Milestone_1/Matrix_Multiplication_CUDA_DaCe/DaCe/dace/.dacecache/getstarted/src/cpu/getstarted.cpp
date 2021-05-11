/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>

void __program_getstarted_internal(double * __restrict__ A, double * __restrict__ __return)
{

    {
        
        
        {
            #pragma omp parallel for
            for (auto __i1 = 0; __i1 < 3; __i1 += 1) {
                for (auto __i0 = 0; __i0 < 2; __i0 += 1) {
                    {
                        double __in1 = A[((3 * __i0) + __i1)];
                        double __in2 = A[((3 * __i0) + __i1)];
                        double __out;

                        ///////////////////
                        // Tasklet code (_Add_)
                        __out = (__in1 + __in2);
                        ///////////////////

                        __return[((3 * __i0) + __i1)] = __out;
                    }
                }
            }
        }
    }
}

DACE_EXPORTED void __program_getstarted(double * __restrict__ A, double * __restrict__ __return)
{
    __program_getstarted_internal(A, __return);
}

DACE_EXPORTED int __dace_init_getstarted(double * __restrict__ A, double * __restrict__ __return)
{
    int __result = 0;

    return __result;
}

DACE_EXPORTED void __dace_exit_getstarted(double * __restrict__ A, double * __restrict__ __return)
{
}

