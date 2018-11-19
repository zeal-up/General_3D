#ifndef _QUERY_BALL_POINT_KERNEL
#define _QUERY_BALL_POINT_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, long *idx, int *pts_cnt, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

