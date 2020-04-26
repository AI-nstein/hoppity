#include "extlib_cuda_kernels.h"
#include <stdexcept>
#include <float.h>

struct SharedMem
{
  __device__ double *getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};


struct Max
{
    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double x, scalar_t y) const {
        return x > static_cast<double>(y) ? x : static_cast<double>(y);
    }
};

struct Add
{
    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double x, scalar_t y) const {
        return x + y;
    }
};


struct SumExp
{
    __device__ __forceinline__ SumExp(double v) : max_k(v) {}

    template<typename scalar_t>
    __device__ __forceinline__ double operator()(double sum, scalar_t v) const {
        return sum + static_cast<double>(exp((double)v - max_k));
    }
    
    const double max_k;
};

template<typename scalar_t>
__global__ void JaggedArgmaxKernel(int64_t* dst, scalar_t *orig_ptr, int64_t* ps)
{
    __shared__ int64_t buffer[256];

    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t cols = ps[blockIdx.x] - ofs;

    scalar_t* row_ptr = orig_ptr + ofs;

    int i_start = threadIdx.x;
    int i_end = cols;
    int i_step = blockDim.x;
    if (i_start < cols)
        buffer[threadIdx.x] = i_start;
    for (int i = i_start + i_step; i < i_end; i += i_step)
    {
      if (row_ptr[i] > row_ptr[buffer[threadIdx.x]])
        buffer[threadIdx.x] = i;
    }
    __syncthreads();

    int shift;
    for (int i = 8 - 1; i >= 0; --i)
    {
    	shift = 1 << i;
    	if (threadIdx.x < shift && threadIdx.x + shift < cols)
    	{
        if (row_ptr[buffer[threadIdx.x + shift]] > row_ptr[buffer[threadIdx.x]])
            buffer[threadIdx.x] = buffer[threadIdx.x + shift];
    	}
		__syncthreads();
    }
    if (threadIdx.x == 0)
    	dst[blockIdx.x] = buffer[0];
}

template<typename scalar_t>
void HostArgmaxForward(scalar_t *input, int64_t *output, int64_t* ps, int64_t bsize)
{
    dim3 grid(bsize);
    dim3 block(256);
    
    JaggedArgmaxKernel<scalar_t><<<grid, block>>>(output, input, ps);
}

template void HostArgmaxForward<float>(float* input, int64_t* output, int64_t* ps, int64_t bsize);
template void HostArgmaxForward<double>(double* input, int64_t* output, int64_t* ps, int64_t bsize);


template<typename scalar_t>
__global__ void Jagged2PaddedKernel(scalar_t* dst, scalar_t *orig_ptr, int64_t* ps, int64_t pad_size)
{
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t cols = ps[blockIdx.x] - ofs;
	scalar_t* src_ptr = orig_ptr + ofs;

    int i_start = threadIdx.x;
    int i_end = cols;
	int i_step = blockDim.x;
	int64_t dst_ofs = blockIdx.x * pad_size;
	scalar_t* dst_ptr = dst + dst_ofs;
    for (int i = i_start; i < i_end; i += i_step)
    {
		dst_ptr[i] = src_ptr[i];
	}
}

template<typename scalar_t>
void HostJagged2PaddedForward(scalar_t *input, scalar_t* output, int64_t* ps, int64_t bsize, int64_t pad_size)
{
    dim3 grid(bsize);
	dim3 block(256);

	Jagged2PaddedKernel<scalar_t><<<grid, block>>>(output, input, ps, pad_size);
}

template void HostJagged2PaddedForward<float>(float *input, float* output, int64_t* ps, int64_t bsize, int64_t pad_size);
template void HostJagged2PaddedForward<double>(double *input, double* output, int64_t* ps, int64_t bsize, int64_t pad_size);


template<typename scalar_t>
__global__ void JaggedAppendForwardKernel(scalar_t* dst, scalar_t *values, scalar_t *suffix, int64_t* ps, int64_t suffix_len)
{
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
	int64_t cols = ps[blockIdx.x] - ofs;
	
	scalar_t* src_val = values + ofs;
	scalar_t* src_suffix = suffix + blockIdx.x * suffix_len;

	scalar_t* dst_ptr = dst + ofs + blockIdx.x * suffix_len;

    int i_start = threadIdx.x;
    int i_end = cols;
	int i_step = blockDim.x;
    for (int i = i_start; i < i_end; i += i_step)
    {
		dst_ptr[i] = src_val[i];
	}
	i_start = threadIdx.x;
	i_end = suffix_len;
	for (int i = i_start; i < i_end; i += i_step)
	{
		dst_ptr[cols + i] = src_suffix[i];
	}
}

template<typename scalar_t>
void HostJaggedAppendForward(scalar_t *values, scalar_t *suffix, scalar_t* output, int64_t* ps, int64_t bsize, int64_t suffix_len)
{
    dim3 grid(bsize);
	dim3 block(256);

	JaggedAppendForwardKernel<scalar_t><<<grid, block>>>(output, values, suffix, ps, suffix_len);
}

template void HostJaggedAppendForward<float>(float *values, float *suffix, float* output, int64_t* ps, int64_t bsize, int64_t suffix_len);
template void HostJaggedAppendForward<double>(double *values, double *suffix, double* output, int64_t* ps, int64_t bsize, int64_t suffix_len);


template<typename scalar_t>
__global__ void JaggedAppendBackwardKernel(scalar_t* gout, scalar_t *grad_val, scalar_t *grad_suffix, int64_t* ps, int64_t suffix_len)
{
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
	int64_t cols = ps[blockIdx.x] - ofs;
	
	scalar_t* dst_val = grad_val + ofs;
	scalar_t* dst_suffix = grad_suffix + blockIdx.x * suffix_len;

	scalar_t* src_ptr = gout + ofs + blockIdx.x * suffix_len;

    int i_start = threadIdx.x;
    int i_end = cols;
	int i_step = blockDim.x;
    for (int i = i_start; i < i_end; i += i_step)
    {
		dst_val[i] = src_ptr[i];
	}
	i_start = threadIdx.x;
	i_end = suffix_len;
	for (int i = i_start; i < i_end; i += i_step)
	{
		dst_suffix[i] = src_ptr[cols + i];
	}
}

template<typename scalar_t>
void HostJaggedAppendBackward(scalar_t *grad_output, scalar_t *grad_val, scalar_t *grad_suffix, int64_t* ps, int64_t bsize, int64_t suffix_len)
{
    dim3 grid(bsize);
	dim3 block(256);

	JaggedAppendBackwardKernel<scalar_t><<<grid, block>>>(grad_output, grad_val, grad_suffix, ps, suffix_len);
}

template void HostJaggedAppendBackward<float>(float *grad_output, float *grad_val, float *grad_suffix, int64_t* ps, int64_t bsize, int64_t suffix_len);
template void HostJaggedAppendBackward<double>(double *grad_output, double *grad_val, double *grad_suffix, int64_t* ps, int64_t bsize, int64_t suffix_len);


template <typename Reduction>
__device__ __forceinline__ double
blockReduce(double* smem, double val,
            const Reduction& r,
            double defaultVal)
{
  // To avoid RaW races from chaining blockReduce calls together, we need a sync here
  __syncthreads();

  smem[threadIdx.x] = val;

  __syncthreads();

  double warpVal = defaultVal;

  // First warp will perform per-warp reductions for the remaining warps
  if (threadIdx.x < 32) {
    int lane = threadIdx.x % 32;
    if (lane < blockDim.x / 32) {
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        warpVal = r(warpVal, smem[lane * 32 + i]);
      }
      smem[lane] = warpVal;
    }
  }

  __syncthreads();

  // First thread will perform a reduction of the above per-warp reductions
  double blockVal = defaultVal;

  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / 32; ++i) {
      blockVal = r(blockVal, smem[i]);
    }
    smem[0] = blockVal;
  }

  // Sync and broadcast
  __syncthreads();
  return smem[0];
}


template <typename Reduction, int ILP, typename scalar_t>
__device__ __forceinline__ double
ilpReduce(scalar_t* data,
          int size,
          const Reduction& r,
          double defaultVal)
{
  double threadVal = defaultVal;
  int offset = threadIdx.x;

  int last = size % (ILP * blockDim.x);

  // Body (unroll by ILP times)
  for (; offset < size - last; offset += blockDim.x * ILP) {
    scalar_t tmp[ILP];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      tmp[j] = data[offset + j * blockDim.x];

#pragma unroll
    for (int j = 0; j < ILP; ++j)
      threadVal = r(threadVal, tmp[j]);
  }

  // Epilogue
  for (; offset < size; offset += blockDim.x)
    threadVal = r(threadVal, data[offset]);

  return threadVal;
}


template <int ILP, typename scalar_t>
__global__ void cunn_SoftMaxForward(scalar_t *output, scalar_t *input, int64_t* ps)
{
    SharedMem smem;
    double *buffer = smem.getPointer();
    // forward pointers to batch[blockIdx.x]
    // each block handles a sample in the mini-batch
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t n_ele = ps[blockIdx.x] - ofs;
    input += ofs;
    output += ofs;
    
    // find the max
    double threadMax = ilpReduce<Max, ILP, scalar_t>(input, n_ele, Max(), -DBL_MAX);
    double max_k = blockReduce<Max>(buffer, threadMax, Max(), -DBL_MAX);
    
    // reduce all values
    double threadExp = ilpReduce<SumExp, ILP, scalar_t>(input, n_ele, SumExp(max_k), static_cast<double>(0));
    
    double sumAll = blockReduce<Add>(buffer, threadExp, Add(), static_cast<double>(0));
    double logsum = max_k + log(sumAll);
    
    int offset = threadIdx.x;
    int last = n_ele % (ILP * blockDim.x);
    for (; offset < n_ele - last; offset += blockDim.x * ILP) {
        scalar_t tmp[ILP];
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            tmp[j] = input[offset + j * blockDim.x];
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            output[offset + j * blockDim.x] = (double)tmp[j] - logsum;
    }
    
    for (; offset < n_ele; offset += blockDim.x)
        output[offset] = (double)input[offset] - logsum;
}


template<typename scalar_t>
void HostLogSoftmaxForward(scalar_t* input, scalar_t *output, int64_t* ps, int64_t bsize)
{
    dim3 grid(bsize);
    dim3 block(1024);
    
    cunn_SoftMaxForward<2>
    <<<grid, block, block.x * sizeof(double)>>>(
        output, input, ps
    );
}

template void HostLogSoftmaxForward<float>(float* input, float* output, int64_t* ps, int64_t bsize);
template void HostLogSoftmaxForward<double>(double* input, double* output, int64_t* ps, int64_t bsize);

template <int ILP, typename scalar_t>
__global__ void cunn_SoftMaxBackward(scalar_t *gradInput, scalar_t *output, scalar_t *gradOutput, int64_t* ps)
{
    SharedMem smem;
    double *buffer = smem.getPointer();
    int64_t ofs = (blockIdx.x == 0) ? 0 : ps[blockIdx.x - 1];
    int64_t n_ele = ps[blockIdx.x] - ofs;
    
    gradInput += ofs;
    output += ofs;
    gradOutput += ofs;
    
    double threadSum = ilpReduce<Add, 4>(gradOutput, n_ele, Add(), double(0));
    double sum_k = blockReduce<Add>(buffer, threadSum, Add(), double(0));
    
    int offset = threadIdx.x;
    int last = n_ele % (ILP * blockDim.x);
    for (; offset < n_ele - last; offset += blockDim.x * ILP) {
        scalar_t tmpGradOutput[ILP];
        scalar_t tmpOutput[ILP];

        #pragma unroll
        for (int j = 0; j < ILP; ++j) {
            tmpGradOutput[j] = gradOutput[offset + j * blockDim.x];
            tmpOutput[j] = output[offset + j * blockDim.x];
        }
        
        #pragma unroll
        for (int j = 0; j < ILP; ++j)
            gradInput[offset + j * blockDim.x] = tmpGradOutput[j] - exp((double)tmpOutput[j]) * sum_k;
    }

    for (; offset < n_ele; offset += blockDim.x)
        gradInput[offset] = gradOutput[offset] - exp((double)output[offset]) * sum_k;
}


template<typename scalar_t>
void HostLogSoftmaxBackward(scalar_t *gradOutput, scalar_t *gradInput, scalar_t *output, int64_t* ps, int64_t bsize)
{
    dim3 grid(bsize);
    dim3 block(1024);
  
    cunn_SoftMaxBackward<2>
    <<<grid, block, block.x * sizeof(double)>>>(
        gradInput, output, gradOutput, ps
    );
}

template void HostLogSoftmaxBackward<float>(float *gradOutput, float *gradInput, float *output, int64_t* ps, int64_t bsize);
template void HostLogSoftmaxBackward<double>(double *gradOutput, double *gradInput, double *output, int64_t* ps, int64_t bsize);


#include <cuda.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif

template<typename scalar_t>
__global__ void NmDistanceKernel(int b,int n,const scalar_t * xyz,int m,const scalar_t * xyz2,scalar_t * result,int64_t * result_i){
	const int batch=512;
	__shared__ scalar_t buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				scalar_t x1=xyz[(i*n+j)*3+0];
				scalar_t y1=xyz[(i*n+j)*3+1];
				scalar_t z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				scalar_t best=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							scalar_t x2=buf[k*3+0]-x1;
							scalar_t y2=buf[k*3+1]-y1;
							scalar_t z2=buf[k*3+2]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							scalar_t x2=buf[k*3+3]-x1;
							scalar_t y2=buf[k*3+4]-y1;
							scalar_t z2=buf[k*3+5]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							scalar_t x2=buf[k*3+6]-x1;
							scalar_t y2=buf[k*3+7]-y1;
							scalar_t z2=buf[k*3+8]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							scalar_t x2=buf[k*3+9]-x1;
							scalar_t y2=buf[k*3+10]-y1;
							scalar_t z2=buf[k*3+11]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							scalar_t x2=buf[k*3+0]-x1;
							scalar_t y2=buf[k*3+1]-y1;
							scalar_t z2=buf[k*3+2]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							scalar_t x2=buf[k*3+3]-x1;
							scalar_t y2=buf[k*3+4]-y1;
							scalar_t z2=buf[k*3+5]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							scalar_t x2=buf[k*3+6]-x1;
							scalar_t y2=buf[k*3+7]-y1;
							scalar_t z2=buf[k*3+8]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							scalar_t x2=buf[k*3+9]-x1;
							scalar_t y2=buf[k*3+10]-y1;
							scalar_t z2=buf[k*3+11]-z1;
							scalar_t d=x2*x2+y2*y2+z2*z2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					scalar_t x2=buf[k*3+0]-x1;
					scalar_t y2=buf[k*3+1]-y1;
					scalar_t z2=buf[k*3+2]-z1;
					scalar_t d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

template<typename scalar_t>
void NmDistanceKernelLauncher(int b,int n,const scalar_t * xyz,int m,const scalar_t * xyz2,scalar_t * result,int64_t * result_i,scalar_t * result2,int64_t * result2_i)
{
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i);
}

template void NmDistanceKernelLauncher<float>(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int64_t * result_i,float * result2,int64_t * result2_i);
template void NmDistanceKernelLauncher<double>(int b,int n,const double * xyz,int m,const double * xyz2,double * result,int64_t * result_i,double * result2,int64_t * result2_i);

template<typename scalar_t>
__global__ void NmDistanceGradKernel(int b,int n,const scalar_t * xyz1,int m,const scalar_t * xyz2,const scalar_t * grad_dist1,const int64_t * idx1,scalar_t * grad_xyz1,scalar_t * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			scalar_t x1=xyz1[(i*n+j)*3+0];
			scalar_t y1=xyz1[(i*n+j)*3+1];
			scalar_t z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			scalar_t x2=xyz2[(i*m+j2)*3+0];
			scalar_t y2=xyz2[(i*m+j2)*3+1];
			scalar_t z2=xyz2[(i*m+j2)*3+2];
			scalar_t g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(z1-z2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*(z1-z2)));
		}
	}
}

template<typename scalar_t>
void NmDistanceGradKernelLauncher(int b,int n,const scalar_t * xyz1,int m,const scalar_t * xyz2,const scalar_t * grad_dist1,const int64_t * idx1,const scalar_t * grad_dist2,const int64_t * idx2,scalar_t * grad_xyz1,scalar_t * grad_xyz2)
{
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);
}


template void NmDistanceGradKernelLauncher<float>(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int64_t * idx1,const float * grad_dist2,const int64_t * idx2,float * grad_xyz1,float * grad_xyz2);
template void NmDistanceGradKernelLauncher<double>(int b,int n,const double * xyz1,int m,const double * xyz2,const double * grad_dist1,const int64_t * idx1,const double * grad_dist2,const int64_t * idx2,double * grad_xyz1,double * grad_xyz2);


template<typename scalar_t>
__global__ void approxmatch(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,scalar_t * __restrict__ match,scalar_t * temp){
	scalar_t * remainL=temp+blockIdx.x*(n+m)*2, * remainR=temp+blockIdx.x*(n+m)*2+n,*ratioL=temp+blockIdx.x*(n+m)*2+n+m,*ratioR=temp+blockIdx.x*(n+m)*2+n+m+n;
	scalar_t multiL,multiR;
	if (n>=m){
		multiL=1;
		multiR=n/m;
	}else{
		multiL=m/n;
		multiR=1;
	}
	const int Block=1024;
	__shared__ scalar_t buf[Block*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x;j<n*m;j+=blockDim.x)
			match[i*n*m+j]=0;
		for (int j=threadIdx.x;j<n;j+=blockDim.x)
			remainL[j]=multiL;
		for (int j=threadIdx.x;j<m;j+=blockDim.x)
			remainR[j]=multiR;
		__syncthreads();
		for (int j=7;j>=-2;j--){
			scalar_t level=-powf(4.0f,j);
			if (j==-2){
				level=0;
			}
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				scalar_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				scalar_t suml=1e-9f;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						scalar_t x2=xyz2[i*m*3+l0*3+l*3+0];
						scalar_t y2=xyz2[i*m*3+l0*3+l*3+1];
						scalar_t z2=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+0]=x2;
						buf[l*4+1]=y2;
						buf[l*4+2]=z2;
						buf[l*4+3]=remainR[l0+l];
					}
					__syncthreads();
					for (int l=0;l<lend;l++){
						scalar_t x2=buf[l*4+0];
						scalar_t y2=buf[l*4+1];
						scalar_t z2=buf[l*4+2];
						scalar_t d=level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						scalar_t w=__expf(d)*buf[l*4+3];
						suml+=w;
					}
					__syncthreads();
				}
				if (k<n)
					ratioL[k]=remainL[k]/suml;
			}
			/*for (int k=threadIdx.x;k<n;k+=gridDim.x){
				scalar_t x1=xyz1[i*n*3+k*3+0];
				scalar_t y1=xyz1[i*n*3+k*3+1];
				scalar_t z1=xyz1[i*n*3+k*3+2];
				scalar_t suml=1e-9f;
				for (int l=0;l<m;l++){
					scalar_t x2=xyz2[i*m*3+l*3+0];
					scalar_t y2=xyz2[i*m*3+l*3+1];
					scalar_t z2=xyz2[i*m*3+l*3+2];
					scalar_t w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*remainR[l];
					suml+=w;
				}
				ratioL[k]=remainL[k]/suml;
			}*/
			__syncthreads();
			for (int l0=0;l0<m;l0+=blockDim.x){
				int l=l0+threadIdx.x;
				scalar_t x2=0,y2=0,z2=0;
				if (l<m){
					x2=xyz2[i*m*3+l*3+0];
					y2=xyz2[i*m*3+l*3+1];
					z2=xyz2[i*m*3+l*3+2];
				}
				scalar_t sumr=0;
				for (int k0=0;k0<n;k0+=Block){
					int kend=min(n,k0+Block)-k0;
					for (int k=threadIdx.x;k<kend;k+=blockDim.x){
						buf[k*4+0]=xyz1[i*n*3+k0*3+k*3+0];
						buf[k*4+1]=xyz1[i*n*3+k0*3+k*3+1];
						buf[k*4+2]=xyz1[i*n*3+k0*3+k*3+2];
						buf[k*4+3]=ratioL[k0+k];
					}
					__syncthreads();
					for (int k=0;k<kend;k++){
						scalar_t x1=buf[k*4+0];
						scalar_t y1=buf[k*4+1];
						scalar_t z1=buf[k*4+2];
						scalar_t w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*buf[k*4+3];
						sumr+=w;
					}
					__syncthreads();
				}
				if (l<m){
					sumr*=remainR[l];
					scalar_t consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
					ratioR[l]=consumption*remainR[l];
					remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
				}
			}
			/*for (int l=threadIdx.x;l<m;l+=blockDim.x){
				scalar_t x2=xyz2[i*m*3+l*3+0];
				scalar_t y2=xyz2[i*m*3+l*3+1];
				scalar_t z2=xyz2[i*m*3+l*3+2];
				scalar_t sumr=0;
				for (int k=0;k<n;k++){
					scalar_t x1=xyz1[i*n*3+k*3+0];
					scalar_t y1=xyz1[i*n*3+k*3+1];
					scalar_t z1=xyz1[i*n*3+k*3+2];
					scalar_t w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k];
					sumr+=w;
				}
				sumr*=remainR[l];
				scalar_t consumption=fminf(remainR[l]/(sumr+1e-9f),1.0f);
				ratioR[l]=consumption*remainR[l];
				remainR[l]=fmaxf(0.0f,remainR[l]-sumr);
			}*/
			__syncthreads();
			for (int k0=0;k0<n;k0+=blockDim.x){
				int k=k0+threadIdx.x;
				scalar_t x1=0,y1=0,z1=0;
				if (k<n){
					x1=xyz1[i*n*3+k*3+0];
					y1=xyz1[i*n*3+k*3+1];
					z1=xyz1[i*n*3+k*3+2];
				}
				scalar_t suml=0;
				for (int l0=0;l0<m;l0+=Block){
					int lend=min(m,l0+Block)-l0;
					for (int l=threadIdx.x;l<lend;l+=blockDim.x){
						buf[l*4+0]=xyz2[i*m*3+l0*3+l*3+0];
						buf[l*4+1]=xyz2[i*m*3+l0*3+l*3+1];
						buf[l*4+2]=xyz2[i*m*3+l0*3+l*3+2];
						buf[l*4+3]=ratioR[l0+l];
					}
					__syncthreads();
					scalar_t rl=ratioL[k];
					if (k<n){
						for (int l=0;l<lend;l++){
							scalar_t x2=buf[l*4+0];
							scalar_t y2=buf[l*4+1];
							scalar_t z2=buf[l*4+2];
							scalar_t w=__expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*rl*buf[l*4+3];
							match[i*n*m+(l0+l)*n+k]+=w;
							suml+=w;
						}
					}
					__syncthreads();
				}
				if (k<n)
					remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}
			/*for (int k=threadIdx.x;k<n;k+=blockDim.x){
				scalar_t x1=xyz1[i*n*3+k*3+0];
				scalar_t y1=xyz1[i*n*3+k*3+1];
				scalar_t z1=xyz1[i*n*3+k*3+2];
				scalar_t suml=0;
				for (int l=0;l<m;l++){
					scalar_t x2=xyz2[i*m*3+l*3+0];
					scalar_t y2=xyz2[i*m*3+l*3+1];
					scalar_t z2=xyz2[i*m*3+l*3+2];
					scalar_t w=expf(level*((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))*ratioL[k]*ratioR[l];
					match[i*n*m+l*n+k]+=w;
					suml+=w;
				}
				remainL[k]=fmaxf(0.0f,remainL[k]-suml);
			}*/
			__syncthreads();
		}
	}
}

template<typename scalar_t>
void approxmatchLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,scalar_t * match,scalar_t * temp)
{
    approxmatch<<<32,512>>>(b,n,m,xyz1,xyz2,match,temp);
}

template void approxmatchLauncher<float>(int b,int n,int m,const float * xyz1,const float * xyz2,float * match,float * temp);
template void approxmatchLauncher<double>(int b,int n,int m,const double * xyz1,const double * xyz2,double * match,double * temp);

template<typename scalar_t>
__global__ void matchcost(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ out){
	__shared__ scalar_t allsum[512];
	const int Block=1024;
	__shared__ scalar_t buf[Block*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		scalar_t subsum=0;
		for (int k0=0;k0<n;k0+=blockDim.x){
			int k=k0+threadIdx.x;
			scalar_t x1=0,y1=0,z1=0;
			if (k<n){
				x1=xyz1[i*n*3+k*3+0];
				y1=xyz1[i*n*3+k*3+1];
				z1=xyz1[i*n*3+k*3+2];
			}
			for (int l0=0;l0<m;l0+=Block){
				int lend=min(m,l0+Block)-l0;
				for (int l=threadIdx.x;l<lend*3;l+=blockDim.x)
					buf[l]=xyz2[i*m*3+l0*3+l];
				__syncthreads();
				if (k<n){
					for (int l=0;l<lend;l++){
						scalar_t x2=buf[l*3+0];
						scalar_t y2=buf[l*3+1];
						scalar_t z2=buf[l*3+2];
						scalar_t d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
						subsum+=d*match[i*n*m+(l0+l)*n+k];
					}
				}
				__syncthreads();
			}
		}
		allsum[threadIdx.x]=subsum;
		for (int j=1;j<blockDim.x;j<<=1){
			__syncthreads();
			if ((threadIdx.x&j)==0 && threadIdx.x+j<blockDim.x){
				allsum[threadIdx.x]+=allsum[threadIdx.x+j];
			}
		}
		if (threadIdx.x==0)
			out[i]=allsum[0];
		__syncthreads();
	}
}

template<typename scalar_t>
void matchcostLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * out)
{
    matchcost<<<32,512>>>(b,n,m,xyz1,xyz2,match,out);
}

template void matchcostLauncher<float>(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out);
template void matchcostLauncher<double>(int b,int n,int m,const double * xyz1,const double * xyz2,const double * match,double * out);

template<typename scalar_t>
__global__ void matchcostgrad2(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad2){
	__shared__ scalar_t sum_grad[256*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		int kbeg=m*blockIdx.y/gridDim.y;
		int kend=m*(blockIdx.y+1)/gridDim.y;
		for (int k=kbeg;k<kend;k++){
			scalar_t x2=xyz2[(i*m+k)*3+0];
			scalar_t y2=xyz2[(i*m+k)*3+1];
			scalar_t z2=xyz2[(i*m+k)*3+2];
			scalar_t subsumx=0,subsumy=0,subsumz=0;
			for (int j=threadIdx.x;j<n;j+=blockDim.x){
				scalar_t x1=x2-xyz1[(i*n+j)*3+0];
				scalar_t y1=y2-xyz1[(i*n+j)*3+1];
				scalar_t z1=z2-xyz1[(i*n+j)*3+2];
				scalar_t d=match[i*n*m+k*n+j]*rsqrtf(fmaxf(x1*x1+y1*y1+z1*z1,1e-20f));
				subsumx+=x1*d;
				subsumy+=y1*d;
				subsumz+=z1*d;
			}
			sum_grad[threadIdx.x*3+0]=subsumx;
			sum_grad[threadIdx.x*3+1]=subsumy;
			sum_grad[threadIdx.x*3+2]=subsumz;
			for (int j=1;j<blockDim.x;j<<=1){
				__syncthreads();
				int j1=threadIdx.x;
				int j2=threadIdx.x+j;
				if ((j1&j)==0 && j2<blockDim.x){
					sum_grad[j1*3+0]+=sum_grad[j2*3+0];
					sum_grad[j1*3+1]+=sum_grad[j2*3+1];
					sum_grad[j1*3+2]+=sum_grad[j2*3+2];
				}
			}
			if (threadIdx.x==0){
				grad2[(i*m+k)*3+0]=sum_grad[0];
				grad2[(i*m+k)*3+1]=sum_grad[1];
				grad2[(i*m+k)*3+2]=sum_grad[2];
			}
			__syncthreads();
		}
	}
}

template<typename scalar_t>
__global__ void matchcostgrad1(int b,int n,int m,const scalar_t * __restrict__ xyz1,const scalar_t * __restrict__ xyz2,const scalar_t * __restrict__ match,scalar_t * __restrict__ grad1){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int l=threadIdx.x;l<n;l+=blockDim.x){
			scalar_t x1=xyz1[i*n*3+l*3+0];
			scalar_t y1=xyz1[i*n*3+l*3+1];
			scalar_t z1=xyz1[i*n*3+l*3+2];
			scalar_t dx=0,dy=0,dz=0;
			for (int k=0;k<m;k++){
				scalar_t x2=xyz2[i*m*3+k*3+0];
				scalar_t y2=xyz2[i*m*3+k*3+1];
				scalar_t z2=xyz2[i*m*3+k*3+2];
				scalar_t d=match[i*n*m+k*n+l]*rsqrtf(fmaxf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2),1e-20f));
				dx+=(x1-x2)*d;
				dy+=(y1-y2)*d;
				dz+=(z1-z2)*d;
			}
			grad1[i*n*3+l*3+0]=dx;
			grad1[i*n*3+l*3+1]=dy;
			grad1[i*n*3+l*3+2]=dz;
		}
	}
}

template<typename scalar_t>
void matchcostgradLauncher(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * grad1,scalar_t * grad2){
	matchcostgrad1<<<32,512>>>(b,n,m,xyz1,xyz2,match,grad1);
	matchcostgrad2<<<dim3(32,32),256>>>(b,n,m,xyz1,xyz2,match,grad2);
}


template void matchcostgradLauncher<float>(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2);
template void matchcostgradLauncher<double>(int b,int n,int m,const double * xyz1,const double * xyz2,const double * match,double * grad1,double * grad2);