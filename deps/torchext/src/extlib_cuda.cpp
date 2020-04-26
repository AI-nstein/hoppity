#include "extlib_cuda.h"
#include "extlib_cuda_kernels.h"
#include <cfloat>
#include <cstdint>

torch::Tensor make_contiguous(torch::Tensor& t)
{
    if (t.is_contiguous())
        return t;
    return t.contiguous();
}

void jagged2padded_forward_cuda(torch::Tensor values, 
                                torch::Tensor prefix_sum, 
                                torch::Tensor padded)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    int64_t* ps = prefix_sum.data<int64_t>();    
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t pad_size = padded.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged2padded_forward_cuda", ([&] {
        HostJagged2PaddedForward(values.data<scalar_t>(), padded.data<scalar_t>(), ps, bsize, pad_size);
    }));
}

void jagged_append_forward_cuda(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor output)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    suffix_mat = make_contiguous(suffix_mat);
    int64_t* ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t suffix_len = suffix_mat.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged_append_forward_cuda", ([&] {
        HostJaggedAppendForward(values.data<scalar_t>(), suffix_mat.data<scalar_t>(), output.data<scalar_t>(), ps, bsize, suffix_len);
    }));
}

void jagged_append_backward_cuda(torch::Tensor grad_output, torch::Tensor prefix_sum, 
                                 torch::Tensor grad_val, torch::Tensor grad_suffix)
{
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    int64_t* ps = prefix_sum.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t suffix_len = grad_suffix.sizes()[1];

    AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "jagged_append_backward_cuda", ([&] {
        HostJaggedAppendBackward(grad_output.data<scalar_t>(), grad_val.data<scalar_t>(), grad_suffix.data<scalar_t>(), 
                                 ps, bsize, suffix_len);
    }));
}


torch::Tensor jagged_argmax_forward_cuda(torch::Tensor values, 
                                         torch::Tensor prefix_sum)
{
    values = make_contiguous(values);
    prefix_sum = make_contiguous(prefix_sum);
    auto output = torch::zeros_like(prefix_sum);
    int64_t* ps = prefix_sum.data<int64_t>();
    int64_t* p_out = output.data<int64_t>();
    int64_t bsize = prefix_sum.sizes()[0];

    AT_DISPATCH_FLOATING_TYPES(values.type(), "jagged_argmax_forward_cuda", ([&] {
        HostArgmaxForward(values.data<scalar_t>(), p_out, ps, bsize);
    }));
    return output;
}


torch::Tensor jagged_log_softmax_forward_cuda(torch::Tensor logits, torch::Tensor prefix_sum)
{
    logits = make_contiguous(logits);
    prefix_sum = make_contiguous(prefix_sum);    
    auto output = torch::zeros_like(logits);
    int64_t bsize = prefix_sum.sizes()[0];
    int64_t* ps = prefix_sum.data<int64_t>();

    AT_DISPATCH_FLOATING_TYPES(logits.type(), "jagged_log_softmax_forward_cuda", ([&] {        
        HostLogSoftmaxForward(logits.data<scalar_t>(),
                              output.data<scalar_t>(),                               
                              ps, bsize);
    }));
    return output;
}

torch::Tensor jagged_log_softmax_backward_cuda(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum)
{
    output = make_contiguous(output);
    grad_output = make_contiguous(grad_output);
    prefix_sum = make_contiguous(prefix_sum);

    auto grad_input = torch::zeros_like(output);

    int64_t bsize = prefix_sum.sizes()[0];
    int64_t* ps = prefix_sum.data<int64_t>();
    AT_DISPATCH_FLOATING_TYPES(output.type(), "jagged_log_softmax_backward_cuda", ([&] {        
        HostLogSoftmaxBackward(grad_output.data<scalar_t>(),
                               grad_input.data<scalar_t>(),
                               output.data<scalar_t>(),
                               ps, bsize);
    }));
    return grad_input;
}

std::vector<torch::Tensor> nn_distance_forward_cuda(torch::Tensor xyz1_tensor, 
                                                    torch::Tensor xyz2_tensor)
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

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "nn_distance_forward_cuda", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* dist1 = dist1_tensor.data<scalar_t>();
        int64_t* idx1 = idx1_tensor.data<int64_t>();
        
        scalar_t* dist2 = dist2_tensor.data<scalar_t>();
        int64_t* idx2 = idx2_tensor.data<int64_t>();
        NmDistanceKernelLauncher(b,n,xyz1,m,xyz2,dist1,idx1,dist2,idx2);
    }));
    return {dist1_tensor, idx1_tensor, dist2_tensor, idx2_tensor};
}

std::vector<torch::Tensor> nn_distance_backward_cuda(torch::Tensor xyz1_tensor, 
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

        NmDistanceGradKernelLauncher(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_dist2,idx2,grad_xyz1,grad_xyz2);        
    }));
    return {grad_xyz1_tensor, grad_xyz2_tensor};
}


torch::Tensor approxmatch_forward_cuda(torch::Tensor xyz1_tensor, 
                                       torch::Tensor xyz2_tensor)
{
    xyz1_tensor = make_contiguous(xyz1_tensor);
    xyz2_tensor = make_contiguous(xyz2_tensor);
    int b = xyz1_tensor.sizes()[0];
    int n = xyz1_tensor.sizes()[1];
    int m = xyz2_tensor.sizes()[1];

    torch::Tensor match_tensor = at::zeros({b, m, n}, xyz1_tensor.options());
    torch::Tensor temp_tensor = at::zeros({b,(n+m)*2}, xyz1_tensor.options());

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "approxmatch_forward_cuda", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        scalar_t* temp = temp_tensor.data<scalar_t>();
        approxmatchLauncher(b,n,m,xyz1,xyz2,match,temp);        
    }));
    return match_tensor;
}

torch::Tensor matchcost_forward_cuda(torch::Tensor xyz1_tensor, 
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

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "matchcost_forward_cuda", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        scalar_t* cost = cost_tensor.data<scalar_t>();

        matchcostLauncher(b,n,m,xyz1,xyz2,match,cost);
    }));
    return cost_tensor;
}

std::vector<torch::Tensor> matchcost_backward_cuda(torch::Tensor xyz1_tensor, 
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

    AT_DISPATCH_FLOATING_TYPES(xyz1_tensor.type(), "matchcost_backward_cuda", ([&]{
        scalar_t* xyz1 = xyz1_tensor.data<scalar_t>(); 
        scalar_t* xyz2 = xyz2_tensor.data<scalar_t>(); 
        
        scalar_t* match = match_tensor.data<scalar_t>();
        scalar_t* grad_xyz1 = grad_xyz1_tensor.data<scalar_t>();
        scalar_t* grad_xyz2 = grad_xyz2_tensor.data<scalar_t>();

        matchcostgradLauncher(b,n,m,xyz1,xyz2,match,grad_xyz1,grad_xyz2);
    }));

    return {grad_xyz1_tensor, grad_xyz2_tensor};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jagged_append_forward_cuda", &jagged_append_forward_cuda, "Jagged Append Forward (CUDA)");
    m.def("jagged_append_backward_cuda", &jagged_append_backward_cuda, "Jagged Append Backward (CUDA)");
    m.def("jagged2padded_forward_cuda", &jagged2padded_forward_cuda, "Pad jagged array (CUDA)");
    m.def("jagged_argmax_forward_cuda", &jagged_argmax_forward_cuda, "Jagged Argmax Forward (CUDA)");
    m.def("jagged_log_softmax_forward_cuda", &jagged_log_softmax_forward_cuda, "Jagged Log Softmax Forward (CUDA)");
    m.def("jagged_log_softmax_backward_cuda", &jagged_log_softmax_backward_cuda, "Jagged Log Softmax Backward (CUDA)");    
    m.def("nn_distance_forward_cuda", &nn_distance_forward_cuda, "NN Distance Forward (CUDA)");
    m.def("nn_distance_backward_cuda", &nn_distance_backward_cuda, "NN Distance Backward (CUDA)");
    m.def("approxmatch_forward_cuda", &approxmatch_forward_cuda, "Approx Match Forward (CUDA)");
    m.def("matchcost_forward_cuda", &matchcost_forward_cuda, "Match Cost Forward (CUDA)");
    m.def("matchcost_backward_cuda", &matchcost_backward_cuda, "Match Cost Backward (CUDA)");    
}