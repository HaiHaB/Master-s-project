#ifndef TESTFUNCTION_H
#define TESTFUNCTION_H


//To test for atom matching for GPU implementation
void testFunction(float *A, float *b, int NA, int NB, int NMax,  const int Min,
        const int Max, const int Incr, float threshold, string fileName,cudaError_t
        (*atomMatchingWithCuda) (float*,  const float *, const float *,  const int,
            const int , const int, float, string));


//To test for calculate simlarity score for GPU implementation
void testFunction(float *A, int NA, int NB, int NMax,  const int Min,
		const int Max, const int Incr, string fileName,cudaError_t
		(*scanMatrixSequentialWithCuda) (float*, const float *,  const int,
				const int , const int, string));

#endif
