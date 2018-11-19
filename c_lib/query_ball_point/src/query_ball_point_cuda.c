#include <THC/THC.h>

#include "query_ball_point_kernel.h"

extern THCState *state;

int query_ball_point_forward_cuda(float radius, int nsample, THCudaTensor * xyz1, THCudaTensor * xyz2, THCudaLongTensor * idx, THCudaIntTensor *pts_cnt)
{
    // Grab the input tensor
    float * xyz1_flat = THCudaTensor_data(state, xyz1);
    float * xyz2_flat = THCudaTensor_data(state, xyz2);
    long * idx_flat = THCudaLongTensor_data(state, idx);
    int * pts_cnt_flat = THCudaIntTensor_data(state, pts_cnt);
    
    int batch_size = THCudaTensor_size(state, xyz1, 0);
    int n = THCudaTensor_size(state, xyz1, 1);
    int m = THCudaTensor_size(state, xyz2, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    queryBallPointLauncher(
        batch_size, n, m, radius, nsample, 
        xyz1_flat,
        xyz2_flat,
        idx_flat,
        pts_cnt_flat,
        stream
        );

    return 1;

}

