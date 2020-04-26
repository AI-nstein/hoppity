#ifndef EXTLIB_CUDA_KERNELS_H
#define EXTLIB_CUDA_KERNELS_H

#include <cstdint>

template<typename scalar_t>
void HostJagged2PaddedForward(scalar_t *input, scalar_t* output, int64_t* ps, int64_t bsize, int64_t pad_size);

template<typename scalar_t>
void HostJaggedAppendForward(scalar_t *values, scalar_t *suffix, scalar_t *output, int64_t* ps, int64_t bsize, int64_t suffix_len);

template<typename scalar_t>
void HostJaggedAppendBackward(scalar_t *grad_output, scalar_t *grad_val, scalar_t *grad_suffix, int64_t* ps, int64_t bsize, int64_t suffix_len);

template<typename scalar_t>
void HostArgmaxForward(scalar_t *input, int64_t* output, int64_t* ps, int64_t bsize);

template<typename scalar_t>
void HostLogSoftmaxForward(scalar_t* input, scalar_t *output, int64_t* ps, int64_t bsize);

template<typename scalar_t>
void HostLogSoftmaxBackward(scalar_t *gradOutput, scalar_t *gradInput, scalar_t *output, int64_t* ps, int64_t bsize);

template<typename scalar_t>
void NmDistanceKernelLauncher(int b,int n,const scalar_t * xyz,int m,const scalar_t * xyz2,scalar_t * result,int64_t * result_i,scalar_t * result2,int64_t * result2_i);

template<typename scalar_t>
void NmDistanceGradKernelLauncher(int b,int n,const scalar_t * xyz1,int m,const scalar_t * xyz2,const scalar_t * grad_dist1,const int64_t * idx1,const scalar_t * grad_dist2,const int64_t * idx2,scalar_t * grad_xyz1,scalar_t * grad_xyz2);

template<typename scalar_t>
void approxmatchLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,scalar_t * match,scalar_t * temp);

template<typename scalar_t>
void matchcostLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * out);

template<typename scalar_t>
void matchcostgradLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * grad1,scalar_t * grad2);

#endif