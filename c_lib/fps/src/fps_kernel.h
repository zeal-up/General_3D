#ifndef _FPS_KERNEL
#define _FPS_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int farthestpointsamplingLauncher(int b, int n, int m, const float * inp,float * temp, long * out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

