#ifndef GRAPHORMER_PREPROCESS_HPP_
#define GRAPHORMER_PREPROCESS_HPP_

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> floyd_warshall_cuda(torch::Tensor adj, const size_t max_dist);

void gen_edge_input_cuda(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features);

#endif  // GRAPHORMER_PREPROCESS_HPP_
