#ifndef EXTLIB_CUDA_H
#define EXTLIB_CUDA_H

#include <torch/extension.h>

void jagged2padded_forward_cuda(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor padded);

void jagged_append_forward_cuda(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor output);

void jagged_append_backward_cuda(torch::Tensor grad_output, torch::Tensor prefix_sum, torch::Tensor grad_val, torch::Tensor grad_suffix);


torch::Tensor jagged_argmax_forward_cuda(torch::Tensor values, torch::Tensor prefix_sum);

torch::Tensor jagged_log_softmax_forward_cuda(torch::Tensor logits, torch::Tensor prefix_sum);

torch::Tensor jagged_log_softmax_backward_cuda(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum);

std::vector<torch::Tensor> nn_distance_forward_cuda(torch::Tensor xyz1_tensor, 
                                                    torch::Tensor xyz2_tensor);

std::vector<torch::Tensor> nn_distance_backward_cuda(torch::Tensor xyz1_tensor, 
                                                     torch::Tensor xyz2_tensor,
                                                     torch::Tensor grad_dist1_tensor,
                                                     torch::Tensor idx1_tensor,
                                                     torch::Tensor grad_dist2_tensor,
                                                     torch::Tensor idx2_tensor);


torch::Tensor approxmatch_forward_cuda(torch::Tensor xyz1_tensor, 
                                       torch::Tensor xyz2_tensor);

torch::Tensor matchcost_forward_cuda(torch::Tensor xyz1_tensor, 
                                     torch::Tensor xyz2_tensor, 
                                     torch::Tensor match_tensor);

std::vector<torch::Tensor> matchcost_backward_cuda(torch::Tensor xyz1_tensor, 
                                                   torch::Tensor xyz2_tensor, 
                                                   torch::Tensor match_tensor);


#endif