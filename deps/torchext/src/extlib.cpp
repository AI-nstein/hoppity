#include "extlib.h"
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <algorithm>

torch::Tensor make_contiguous(torch::Tensor& t)
{
    if (t.is_contiguous())
        return t;
    return t.contiguous();
}

template<typename scalar_t>
void impl_jagged_argmax_forward(scalar_t* input_data_base, 
                                torch::Tensor prefix_sum, 
                                torch::Tensor& output)
{
    int64_t* ps = prefix_sum.data<int64_t>();
    int64_t* p_out = output.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t i, d;

    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        int64_t n_ele = ps[i] - offset;
        if (n_ele)
        {
            scalar_t* input_data  = input_data_base  + offset;
            
            scalar_t max_input = input_data[0];
            int64_t max_id = 0;
            for (d = 1; d < n_ele; d++)
                if (input_data[d] > max_input)
                {
                    max_input = input_data[d];
                    max_id = d;
                }
                p_out[i] = max_id;
        }
    }
}

template void impl_jagged_argmax_forward<float>(float* input_data_base, torch::Tensor prefix_sum, torch::Tensor& output);
template void impl_jagged_argmax_forward<double>(double* input_data_base, torch::Tensor prefix_sum, torch::Tensor& output);

torch::Tensor jagged_argmax_forward(torch::Tensor values, 
                                    torch::Tensor prefix_sum)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    auto output = torch::zeros_like(prefix_sum);
    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged_argmax_forward", ([&] {
        impl_jagged_argmax_forward(values.data<scalar_t>(), 
                                   prefix_sum,
                                   output);
    }));
    return output;
}

template<typename scalar_t>
void impl_jagged_append_forward(scalar_t* input_data_base, 
                                torch::Tensor prefix_sum,
                                torch::Tensor suffix_mat, 
                                torch::Tensor& output)
{
    int64_t* ps = prefix_sum.data<int64_t>();
    scalar_t* p_out = output.data<scalar_t>();
    scalar_t* p_suffix = suffix_mat.data<scalar_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t suffix_len = suffix_mat.sizes()[1];

    #pragma omp parallel for
    for (int64_t i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        int64_t dst_offset = offset + i * suffix_len;
        int64_t n_ele = ps[i] - offset;

        scalar_t* input_data = input_data_base + offset;
        scalar_t* out_ptr = p_out + dst_offset;
        if (n_ele)
            std::memcpy(out_ptr, input_data, n_ele * sizeof(scalar_t));
        out_ptr = out_ptr + n_ele;
        input_data = p_suffix + i * suffix_len;
        std::memcpy(out_ptr, input_data, suffix_len * sizeof(scalar_t));
    }    
}

template void impl_jagged_append_forward<float>(float* input_data_base, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor& output);
template void impl_jagged_append_forward<double>(double* input_data_base, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor& output);


void jagged_append_forward(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor output)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    suffix_mat = make_contiguous(suffix_mat);
    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged_append_forward", ([&] {
        impl_jagged_append_forward(values.data<scalar_t>(), 
                                   prefix_sum,
                                   suffix_mat,
                                   output);
    }));
}


template<typename scalar_t>
void impl_jagged2padded_forward(scalar_t* input_data_base, 
                                torch::Tensor prefix_sum, 
                                torch::Tensor& output)
{
    int64_t* ps = prefix_sum.data<int64_t>();
    scalar_t* p_out = output.data<scalar_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t pad_size = output.sizes()[1];

    #pragma omp parallel for
    for (int64_t i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        int64_t n_ele = ps[i] - offset;
        if (n_ele)
        {
            scalar_t* input_data  = input_data_base  + offset;
            scalar_t* out_ptr = p_out + i * pad_size;
            std::memcpy(out_ptr, input_data, n_ele * sizeof(scalar_t));
        }
    }
}

template void impl_jagged2padded_forward<float>(float* input_data_base, torch::Tensor prefix_sum, torch::Tensor& output);
template void impl_jagged2padded_forward<double>(double* input_data_base, torch::Tensor prefix_sum, torch::Tensor& output);

void jagged2padded_forward(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor padded)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged2padded_forward", ([&] {
        impl_jagged2padded_forward(values.data<scalar_t>(), 
                                   prefix_sum,
                                   padded);
    }));    
}


template<typename scalar_t>
void impl_jagged_log_softmax_forward(scalar_t *input_data_base, scalar_t *output_data_base, torch::Tensor prefix_sum)
{
    int64_t *ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t i, d;
    
    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        
        scalar_t* input_data  = input_data_base  + offset;
        scalar_t* output_data = output_data_base + offset;
        
        int64_t n_ele = ps[i] - offset;
        scalar_t max_input = -FLT_MAX;
        
        for (d = 0; d < n_ele; d++)
            max_input = std::max(max_input, input_data[d]);
            
        double logsum = 0;
        for (d = 0; d < n_ele; d++)
            logsum += exp(input_data[d] - max_input);
        logsum = max_input + log(logsum);
        for (d = 0; d < n_ele; d++)
            output_data[d] = input_data[d] - logsum;
    }
}


template void impl_jagged_log_softmax_forward<float>(float *input_data_base, float *output_data_base, torch::Tensor prefix_sum);
template void impl_jagged_log_softmax_forward<double>(double *input_data_base, double *output_data_base, torch::Tensor prefix_sum);


torch::Tensor jagged_log_softmax_forward(torch::Tensor logits, torch::Tensor prefix_sum)
{
    logits = make_contiguous(logits);
    prefix_sum = make_contiguous(prefix_sum);
    auto output = torch::zeros_like(logits);
    AT_DISPATCH_FLOATING_TYPES(logits.type(), "jagged_log_softmax_forward", ([&] {
        impl_jagged_log_softmax_forward(logits.data<scalar_t>(), 
                                        output.data<scalar_t>(),                                         
                                        prefix_sum);
    }));
    return output;
}

template<typename scalar_t>
void impl_jagged_log_softmax_backward(scalar_t *output_data_base, scalar_t *gradOutput_data_base, torch::Tensor prefix_sum, scalar_t *gradInput_data_base)
{
    int64_t *ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t i, d;

    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        scalar_t *gradInput_data  = gradInput_data_base  + offset;
        scalar_t *output_data     = output_data_base     + offset;
        scalar_t *gradOutput_data = gradOutput_data_base + offset;
        
        double sum = 0;
        int64_t n_ele = ps[i] - offset;
        for (d = 0; d < n_ele; d++)
            sum += gradOutput_data[d];
        
        for (d = 0; d < n_ele; d++)
            gradInput_data[d] = gradOutput_data[d] - exp(output_data[d]) * sum;
    }
}

template void impl_jagged_log_softmax_backward<float>(float *output_data_base, float *gradOutput_data_base, torch::Tensor prefix_sum, float *gradInput_data_base);
template void impl_jagged_log_softmax_backward<double>(double *output_data_base, double *gradOutput_data_base, torch::Tensor prefix_sum, double *gradInput_data_base);

torch::Tensor jagged_log_softmax_backward(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum)
{
    output = make_contiguous(output);
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    auto grad_input = torch::zeros_like(output);

    AT_DISPATCH_FLOATING_TYPES(output.type(), "jagged_log_softmax_backward", ([&] {
        impl_jagged_log_softmax_backward(output.data<scalar_t>(), 
                                         grad_output.data<scalar_t>(), 
                                         prefix_sum,
                                         grad_input.data<scalar_t>());
    }));

    return grad_input;
}

template<typename scalar_t>
void impl_jagged_append_backward(scalar_t* gradout_base, torch::Tensor prefix_sum, torch::Tensor& grad_val, torch::Tensor& grad_suffix)
{
    int64_t* ps = prefix_sum.data<int64_t>();
    scalar_t* p_out_val = grad_val.data<scalar_t>();
    scalar_t* p_out_suffix = grad_suffix.data<scalar_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t suffix_len = grad_suffix.sizes()[1];

    #pragma omp parallel for
    for (int64_t i = 0; i < bsize; i++)
    {
        int64_t offset = (i == 0) ? 0 : ps[i - 1];
        int64_t dst_offset = offset + i * suffix_len;
        int64_t n_ele = ps[i] - offset;

        scalar_t* val_data = p_out_val + offset;
        scalar_t* grad_out_ptr = gradout_base + dst_offset;
        if (n_ele)
            std::memcpy(val_data, grad_out_ptr, n_ele * sizeof(scalar_t));
        grad_out_ptr = grad_out_ptr + n_ele;
        scalar_t* suffix_data = p_out_suffix + i * suffix_len;
        std::memcpy(suffix_data, grad_out_ptr, suffix_len * sizeof(scalar_t));
    }
}

template void impl_jagged_append_backward<float>(float* gradout_base, torch::Tensor prefix_sum, torch::Tensor& grad_val, torch::Tensor& grad_suffix);
template void impl_jagged_append_backward<double>(double* gradout_base, torch::Tensor prefix_sum, torch::Tensor& grad_val, torch::Tensor& grad_suffix);

void jagged_append_backward(torch::Tensor grad_output, torch::Tensor prefix_sum, torch::Tensor grad_val, torch::Tensor grad_suffix)
{
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "jagged_append_backward", ([&] {
        impl_jagged_append_backward(grad_output.data<scalar_t>(), 
                                    prefix_sum,
                                    grad_val,
                                    grad_suffix);
    }));
}

template<typename scalar_t>
static void nnsearch(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,scalar_t * dist, int64_t * idx){
	for (int64_t i=0;i<b;i++){
		for (int64_t j=0;j<n;j++){
			scalar_t x1=xyz1[(i*n+j)*3+0];
			scalar_t y1=xyz1[(i*n+j)*3+1];
			scalar_t z1=xyz1[(i*n+j)*3+2];
			double best=0;
			int64_t besti=0;
			for (int64_t k=0;k<m;k++){
				scalar_t x2=xyz2[(i*m+k)*3+0]-x1;
				scalar_t y2=xyz2[(i*m+k)*3+1]-y1;
				scalar_t z2=xyz2[(i*m+k)*3+2]-z1;
				double d=x2*x2+y2*y2+z2*z2;
				if (k==0 || d<best){
					best=d;
					besti=k;
				}
			}
			dist[i*n+j]=best;
			idx[i*n+j]=besti;
		}
	}
}

template void nnsearch<float>(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int64_t * idx);
template void nnsearch<double>(int b,int n,int m,const double * xyz1,const double * xyz2,double * dist,int64_t * idx);

std::vector<torch::Tensor> nn_distance_forward(torch::Tensor xyz1_tensor, torch::Tensor xyz2_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];
    
    torch::Tensor dist1_tensor = at::zeros({b, n}, xyz1_tensor.options());
    torch::Tensor idx1_tensor = at::zeros({b, n}, xyz1_tensor.options().dtype(at::kLong).requires_grad(false));
    
    torch::Tensor dist2_tensor = at::zeros({b, m}, xyz2_tensor.options());
    torch::Tensor idx2_tensor = at::zeros({b, m}, xyz2_tensor.options().dtype(at::kLong).requires_grad(false));
    
    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "nn_distance_forward", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* dist1 = dist1_tensor.data<scalar_t>();
        int64_t* idx1 = idx1_tensor.data<int64_t>();
        nnsearch(b,n,m,xyz1,xyz2,dist1,idx1);
        
        scalar_t* dist2 = dist2_tensor.data<scalar_t>();
        int64_t* idx2 = idx2_tensor.data<int64_t>();
        nnsearch(b,m,n,xyz2,xyz1,dist2,idx2);
    }));

    return {dist1_tensor, idx1_tensor, dist2_tensor, idx2_tensor};
}


std::vector<torch::Tensor> nn_distance_backward(torch::Tensor xyz1_tensor, 
                                                torch::Tensor xyz2_tensor,
                                                torch::Tensor grad_dist1_tensor,
                                                torch::Tensor idx1_tensor,
                                                torch::Tensor grad_dist2_tensor,
                                                torch::Tensor idx2_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    grad_dist1_tensor = make_contiguous(grad_dist1_tensor);
    idx1_tensor = make_contiguous(idx1_tensor);
    grad_dist2_tensor = make_contiguous(grad_dist2_tensor);
    idx2_tensor = make_contiguous(idx2_tensor);

    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];

    torch::Tensor grad_xyz1_tensor = at::zeros_like(xyz1_tensor);
    torch::Tensor grad_xyz2_tensor = at::zeros_like(xyz2_tensor);

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "nn_distance_backward", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* grad_dist1 = grad_dist1_tensor.data<scalar_t>();
        int64_t* idx1 = idx1_tensor.data<int64_t>();
        scalar_t* grad_dist2 = grad_dist2_tensor.data<scalar_t>();
        int64_t* idx2 = idx2_tensor.data<int64_t>();

        scalar_t* grad_xyz1 = grad_xyz1_tensor.data<scalar_t>();
        scalar_t* grad_xyz2 = grad_xyz2_tensor.data<scalar_t>();
        for (int i=0;i<b;i++){
            for (int j=0;j<n;j++){
                scalar_t x1=xyz1[(i*n+j)*3+0];
                scalar_t y1=xyz1[(i*n+j)*3+1];
                scalar_t z1=xyz1[(i*n+j)*3+2];
                int64_t j2=idx1[i*n+j];
                scalar_t x2=xyz2[(i*m+j2)*3+0];
                scalar_t y2=xyz2[(i*m+j2)*3+1];
                scalar_t z2=xyz2[(i*m+j2)*3+2];
                scalar_t g=grad_dist1[i*n+j]*2;
                grad_xyz1[(i*n+j)*3+0]+=g*(x1-x2);                
                grad_xyz1[(i*n+j)*3+1]+=g*(y1-y2);
                grad_xyz1[(i*n+j)*3+2]+=g*(z1-z2);
                grad_xyz2[(i*m+j2)*3+0]-=(g*(x1-x2));
                grad_xyz2[(i*m+j2)*3+1]-=(g*(y1-y2));
                grad_xyz2[(i*m+j2)*3+2]-=(g*(z1-z2));
            }
            for (int j=0;j<m;j++){
                scalar_t x1=xyz2[(i*m+j)*3+0];
                scalar_t y1=xyz2[(i*m+j)*3+1];
                scalar_t z1=xyz2[(i*m+j)*3+2];
                int64_t j2=idx2[i*m+j];
                scalar_t x2=xyz1[(i*n+j2)*3+0];
                scalar_t y2=xyz1[(i*n+j2)*3+1];
                scalar_t z2=xyz1[(i*n+j2)*3+2];
                scalar_t g=grad_dist2[i*m+j]*2;
                grad_xyz2[(i*m+j)*3+0]+=g*(x1-x2);
                grad_xyz2[(i*m+j)*3+1]+=g*(y1-y2);
                grad_xyz2[(i*m+j)*3+2]+=g*(z1-z2);
                grad_xyz1[(i*n+j2)*3+0]-=(g*(x1-x2));
                grad_xyz1[(i*n+j2)*3+1]-=(g*(y1-y2));
                grad_xyz1[(i*n+j2)*3+2]-=(g*(z1-z2));
            }
        }
    }));
    return {grad_xyz1_tensor, grad_xyz2_tensor};
}

template<typename scalar_t>
void approxmatch_cpu(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,scalar_t * match){
	for (int i=0;i<b;i++){
		int factorl=std::max(n,m)/n;
		int factorr=std::max(n,m)/m;
		std::vector<double> saturatedl(n,double(factorl)),saturatedr(m,double(factorr));
		std::vector<double> weight(n*m);
		for (int j=0;j<n*m;j++)
			match[j]=0;
		for (int j=8;j>=-2;j--){
			//printf("i=%d j=%d\n",i,j);
			double level=-powf(4.0,j);
			if (j==-2)
				level=0;
			for (int k=0;k<n;k++){
				double x1=xyz1[k*3+0];
				double y1=xyz1[k*3+1];
				double z1=xyz1[k*3+2];
				for (int l=0;l<m;l++){
					double x2=xyz2[l*3+0];
					double y2=xyz2[l*3+1];
					double z2=xyz2[l*3+2];
					weight[k*m+l]=expf(level*((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2)))*saturatedr[l];
				}
			}
			std::vector<double> ss(m,1e-9);
			for (int k=0;k<n;k++){
				double s=1e-9;
				for (int l=0;l<m;l++){
					s+=weight[k*m+l];
				}
				for (int l=0;l<m;l++){
					weight[k*m+l]=weight[k*m+l]/s*saturatedl[k];
				}
				for (int l=0;l<m;l++)
					ss[l]+=weight[k*m+l];
			}
			for (int l=0;l<m;l++){
				double s=ss[l];
				double r=std::min(saturatedr[l]/s,1.0);
				ss[l]=r;
			}
			std::vector<double> ss2(m,0);
			for (int k=0;k<n;k++){
				double s=0;
				for (int l=0;l<m;l++){
					weight[k*m+l]*=ss[l];
					s+=weight[k*m+l];
					ss2[l]+=weight[k*m+l];
				}
				saturatedl[k]=std::max(saturatedl[k]-s,0.0);
			}
			for (int k=0;k<n*m;k++)
				match[k]+=weight[k];
			for (int l=0;l<m;l++){
				saturatedr[l]=std::max(saturatedr[l]-ss2[l],0.0);
			}
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
	}
}

template void approxmatch_cpu<float>(int b,int n,int m,const float * xyz1,const float * xyz2,float * match);
template void approxmatch_cpu<double>(int b,int n,int m,const double * xyz1,const double * xyz2,double * match);

torch::Tensor approxmatch_forward(torch::Tensor xyz1_tensor, 
                                  torch::Tensor xyz2_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];

    torch::Tensor match_tensor = at::zeros({b, m, n}, xyz1_tensor.options());
    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "approxmatch_forward", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        approxmatch_cpu(b,n,m,xyz1,xyz2,match);
    }));
    return match_tensor;
}


template<typename scalar_t>
void matchcost_cpu(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * cost){
	for (int i=0;i<b;i++){
		double s=0;
		for (int j=0;j<n;j++)
			for (int k=0;k<m;k++){
				scalar_t x1=xyz1[j*3+0];
				scalar_t y1=xyz1[j*3+1];
				scalar_t z1=xyz1[j*3+2];
				scalar_t x2=xyz2[k*3+0];
				scalar_t y2=xyz2[k*3+1];
				scalar_t z2=xyz2[k*3+2];
				scalar_t d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1))*match[j*m+k];
				s+=d;
			}
		cost[0]=s;
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		cost+=1;
	}
}

template void matchcost_cpu<float>(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * cost);
template void matchcost_cpu<double>(int b,int n,int m,const double * xyz1,const double * xyz2,const double * match,double * cost);


torch::Tensor matchcost_forward(torch::Tensor xyz1_tensor, 
                                torch::Tensor xyz2_tensor, 
                                torch::Tensor match_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    match_tensor = make_contiguous(match_tensor);

    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];

    torch::Tensor cost_tensor = at::zeros({b}, xyz1_tensor.options());
    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "matchcost_forward", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        scalar_t* cost = cost_tensor.data<scalar_t>();

        matchcost_cpu(b,n,m,xyz1,xyz2,match,cost);
    }));

    return cost_tensor;
}


template<typename scalar_t>
void matchcostgrad_cpu(int b,int n,int m,const scalar_t * xyz1,const scalar_t * xyz2,const scalar_t * match,scalar_t * grad1,scalar_t * grad2){
	for (int i=0;i<b;i++){
		for (int j=0;j<n;j++)
			grad1[j*3+0]=0;
		for (int j=0;j<m;j++){
			scalar_t sx=0,sy=0,sz=0;
			for (int k=0;k<n;k++){
				scalar_t x2=xyz2[j*3+0];
				scalar_t y2=xyz2[j*3+1];
				scalar_t z2=xyz2[j*3+2];
				scalar_t x1=xyz1[k*3+0];
				scalar_t y1=xyz1[k*3+1];
				scalar_t z1=xyz1[k*3+2];
				scalar_t d=std::max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
				scalar_t dx=match[k*m+j]*((x2-x1)/d);
				scalar_t dy=match[k*m+j]*((y2-y1)/d);
				scalar_t dz=match[k*m+j]*((z2-z1)/d);
				grad1[k*3+0]-=dx;
				grad1[k*3+1]-=dy;
				grad1[k*3+2]-=dz;
				sx+=dx;
				sy+=dy;
				sz+=dz;
			}
			grad2[j*3+0]=sx;
			grad2[j*3+1]=sy;
			grad2[j*3+2]=sz;
		}
		xyz1+=n*3;
		xyz2+=m*3;
		match+=n*m;
		grad1+=n*3;
		grad2+=m*3;
	}
}


template void matchcostgrad_cpu<float>(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2);
template void matchcostgrad_cpu<double>(int b,int n,int m,const double * xyz1,const double * xyz2,const double * match,double * grad1,double * grad2);

std::vector<torch::Tensor> matchcost_backward(torch::Tensor xyz1_tensor, 
                                              torch::Tensor xyz2_tensor, 
                                              torch::Tensor match_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    match_tensor = make_contiguous(match_tensor);

    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];

    torch::Tensor grad_xyz1_tensor = at::zeros_like(xyz1_tensor);
    torch::Tensor grad_xyz2_tensor = at::zeros_like(xyz2_tensor);

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "matchcost_backward", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        scalar_t* grad_xyz1 = grad_xyz1_tensor.data<scalar_t>();
        scalar_t* grad_xyz2 = grad_xyz2_tensor.data<scalar_t>();

        matchcostgrad_cpu(b,n,m,xyz1,xyz2,match,grad_xyz1,grad_xyz2);
    }));

    return {grad_xyz1_tensor, grad_xyz2_tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("jagged_append_forward", &jagged_append_forward, "Append jagged array with mat");
  m.def("jagged2padded_forward", &jagged2padded_forward, "Pad jagged array");
  m.def("jagged_argmax_forward", &jagged_argmax_forward, "Jagged Argmax Forward");
  m.def("jagged_log_softmax_forward", &jagged_log_softmax_forward, "Jagged Log Softmax Forward");
  m.def("jagged_append_backward", &jagged_append_backward, "Jagged Append Backward"); 
  m.def("jagged_log_softmax_backward", &jagged_log_softmax_backward, "Jagged Log Softmax Backward");  
  m.def("nn_distance_forward", &nn_distance_forward, "NN Distance Forward");
  m.def("nn_distance_backward", &nn_distance_backward, "NN Distance Backward");
  m.def("approxmatch_forward", &approxmatch_forward, "Approx Match Forward");
  m.def("matchcost_forward", &matchcost_forward, "Match Cost Forward");
  m.def("matchcost_backward", &matchcost_backward, "Match Cost Backward");
}
