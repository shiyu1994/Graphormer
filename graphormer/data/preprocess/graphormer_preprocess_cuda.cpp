#include "graphormer_preprocess.hpp"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> floyd_warshall(torch::Tensor adj, size_t max_dist) {
  CHECK_CUDA(adj);
  return floyd_warshall_cuda(adj, max_dist);
}

void gen_edge_input(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  CHECK_CUDA(pred);
  CHECK_CUDA(dist);
  CHECK_CUDA(edge_features);
  CHECK_CUDA(output_edge_features);
  gen_edge_input_cuda(max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("floyd_warshall", &floyd_warshall, "Floyd Warshall (CUDA)");
  m.def("gen_edge_input", &gen_edge_input, "Get Edge Input (CUDA)");
}
