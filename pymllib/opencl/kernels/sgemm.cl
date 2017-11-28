// SGEMM Routines
//
//

__kernel void sgemm_naive(const int M, const int N, const int K, 
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

#define TILE16 16
__kernel void sgemm_tiling16(const int M, const int N, const int K,
        const __global float* A,
        const __global float* B,
              __global float* C)
{
    // Get thread idx 
    const int row = get_local_id(0);        // max = 32
    const int col = get_local_id(1);        // max = 32
    const int global_row = TILE16 * get_group_id(0) + row;
    const int global_col = TILE16 * get_group_id(1) + col;

    // Local memory for a tile of size TILE16 x TILE16
    __local float A_SUB[TILE16][TILE16];
    __local float B_SUB[TILE16][TILE16];

    // Init the accumulation register
    float acc = 0.0f;
    // Loop over the tiles 
    int t, k;
    const int num_tiles = K / TILE16;
    for(t = 0; t < num_tiles; t++)
    {
        // load one tile of A and B into local memory 
        const int tiled_row = TILE16 * t + row;
        const int tiled_col = TILE16 * t + col;
        // Copy memory and synchronize 
        A_SUB[col][row] = A[tiled_col  * M + global_row];
        B_SUB[col][row] = B[global_col * K + tiled_row];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the result for a single tile 
        for(k = 0; k < TILE16; k++)
        {
           acc += A_SUB[k][row] * B_SUB[col][k]; 
        }
        barrier(CLK_LOCAL_MEM_FENCE);       // sync this tile
    }

    // Store final result 
    C[global_col * M + global_row] = acc;
}

// SGEMM with tiling in local memory 
// Tile size for this kernel is fixed at 32
//
#define TILE32 32
__kernel void sgemm_tiling32(const int M, const int N, const int K,
        const __global float* A,
        const __global float* B,
              __global float* C)
{
    // Get thread idx 
    const int row = get_local_id(0);        // max = 32
    const int col = get_local_id(1);        // max = 32
    const int global_row = TILE32 * get_group_id(0) + row;
    const int global_col = TILE32 * get_group_id(1) + col;

    // Local memory for a tile of size TILE32 x TILE32
    __local float A_SUB[TILE32][TILE32];
    __local float B_SUB[TILE32][TILE32];

    // Init the accumulation register
    float acc = 0.0f;
    // Loop over the tiles 
    int t, k;
    const int num_tiles = K / TILE32;
    for(t = 0; t < num_tiles; t++)
    {
        // load one tile of A and B into local memory 
        const int tiled_row = TILE32 * t + row;
        const int tiled_col = TILE32 * t + col;
        // Copy memory and synchronize 
        A_SUB[col][row] = A[tiled_col  * M + global_row];
        B_SUB[col][row] = B[global_col * K + tiled_row];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute the result for a single tile 
        for(k = 0; k < TILE32; k++)
        {
           acc += A_SUB[k][row] * B_SUB[col][k]; 
        }
        barrier(CLK_LOCAL_MEM_FENCE);       // sync this tile
    }

    // Store final result 
    C[global_col * M + global_row] = acc;
}



