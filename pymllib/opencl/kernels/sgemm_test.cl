// SGEMM Routines
//
// The first one here is just a naive one for testing the OpenCL routines
//

__kernel void GEMM1(const int M, const int N, const int K, 
        const __global float* A, 
        const __global float* B,
        __global float* C)
{
    // Thread idx
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    // loop over k and compute a single element
    float acc = 0.0f;
    for(int k = 0; k < K; k++)
    {
        acc += A[k*M + global_row] * B[global_col * K + k];
    }

    // Store result
    C[global_col * M + global_row] = acc;
}
