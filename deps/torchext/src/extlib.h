#ifndef EXTLIB_H
#define EXTLIB_H

#include <torch/extension.h>
#include <vector>


void jagged2padded_forward(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor padded);

torch::Tensor jagged_argmax_forward(torch::Tensor values, torch::Tensor prefix_sum);

torch::Tensor jagged_log_softmax_forward(torch::Tensor logits, torch::Tensor prefix_sum);

void jagged_append_forward(torch::Tensor values, torch::Tensor prefix_sum, torch::Tensor suffix_mat, torch::Tensor output);

void jagged_append_backward(torch::Tensor grad_output, torch::Tensor prefix_sum, torch::Tensor grad_val, torch::Tensor grad_suffix);

torch::Tensor jagged_log_softmax_backward(torch::Tensor output, torch::Tensor grad_output, torch::Tensor prefix_sum);


std::vector<torch::Tensor> nn_distance_forward(torch::Tensor xyz1_tensor, 
                                               torch::Tensor xyz2_tensor);

std::vector<torch::Tensor> nn_distance_backward(torch::Tensor xyz1_tensor, 
                                                torch::Tensor xyz2_tensor,
                                                torch::Tensor grad_dist1_tensor,
                                                torch::Tensor idx1_tensor,
                                                torch::Tensor grad_dist2_tensor,
                                                torch::Tensor idx2_tensor);

torch::Tensor approxmatch_forward(torch::Tensor xyz1_tensor, 
                                  torch::Tensor xyz2_tensor);

torch::Tensor matchcost_forward(torch::Tensor xyz1_tensor, 
                                torch::Tensor xyz2_tensor, 
                                torch::Tensor match_tensor);

std::vector<torch::Tensor> matchcost_backward(torch::Tensor xyz1_tensor, 
                                              torch::Tensor xyz2_tensor, 
                                              torch::Tensor match_tensor);


#endif
