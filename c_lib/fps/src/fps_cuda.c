#include <THC/THC.h>

#include "fps_kernel.h"

extern THCState *state;

int fps_forward_cuda(int num_sample_points, THCudaTensor * input, THCudaTensor *temp, THCudaLongTensor * output)
{
    // Grab the input tensor
    float * input_flat = THCudaTensor_data(state, input);
    float * temp_flat = THCudaTensor_data(state, temp);
    long * output_flat = THCudaLongTensor_data(state, output);

    int batch_size = THCudaTensor_size(state, input, 0);
    int num_points = THCudaTensor_size(state, input, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    farthestpointsamplingLauncher(
        batch_size, num_points, num_sample_points,
        input_flat,
        temp_flat,
        output_flat,
        stream
        );

    return 1;

}

