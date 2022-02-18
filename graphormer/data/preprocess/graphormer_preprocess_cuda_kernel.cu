#include "graphormer_preprocess.hpp"

#define FLOYD_WARSHALL_BLOCK_DIM_X (16)
#define FLOYD_WARSHALL_BLOCK_DIM_Y (16)

template <typename int_t>
__global__ void floyd_warshall_set_init_kernel(
  int_t num_nodes,
  int_t max_dist,
  const torch::PackedTensorAccessor64<int_t, 2> adj,
  torch::PackedTensorAccessor64<int_t, 2> output_dist) {
  const int_t i = static_cast<int_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int_t j = static_cast<int_t>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes) {
    if (i == j) {
      output_dist[i][j] = 0;
    } else {
      if (adj[i][j]) {
        output_dist[i][j] = 1;
      } else {
        output_dist[i][j] = max_dist;
      }
    }
  }
}

template <typename int_t>
__global__ void floyd_warshall_cuda_one_iter_kernel(
  int_t num_nodes,
  torch::PackedTensorAccessor64<int_t, 2> output_dist,
  torch::PackedTensorAccessor64<int_t, 2> output_pred,
  int_t k) {
  const int_t i = static_cast<int_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int_t j = static_cast<int_t>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes && i != k && j != k && i != j) {
    const int_t new_dist = output_dist[i][k] + output_dist[k][j];
    const int_t old_dist = output_dist[i][j];
    if (new_dist < old_dist) {
      output_dist[i][j] = new_dist;
      output_pred[i][j] = k;
    }
  }
}

std::vector<torch::Tensor> floyd_warshall_cuda(const torch::Tensor adj, const size_t max_dist) {
  const size_t num_nodes = adj.size(0);
  const size_t grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const size_t grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  auto output_dist = torch::zeros_like(adj);
  auto output_pred = torch::zeros_like(adj);
  AT_DISPATCH_INTEGRAL_TYPES(
    adj.type(), "floyd_warshall_set_init_kernel", ([&] {
      floyd_warshall_set_init_kernel<scalar_t><<<grid_dim, block_dim>>>(
        num_nodes,
        max_dist,
        adj.packed_accessor64<scalar_t, 2>(),
        output_dist.packed_accessor64<scalar_t, 2>());
  }));
  for (size_t k = 0; k < num_nodes; ++k) {
    AT_DISPATCH_INTEGRAL_TYPES(
      adj.type(), "floyd_warshall_cuda_one_iter_kernel", ([&] {
        floyd_warshall_cuda_one_iter_kernel<scalar_t><<<grid_dim, block_dim>>>(
          num_nodes,
          output_dist.packed_accessor64<scalar_t, 2>(),
          output_pred.packed_accessor64<scalar_t, 2>(),
          k);
    }));
  }
  return {output_dist, output_pred};
}

template <typename int_t, size_t MAX_DIST>
__device__ void get_path(
  const int_t num_nodes,
  const int_t max_dist,
  const int_t i,
  const int_t j,
  const torch::PackedTensorAccessor64<int_t, 2> pred,
  const torch::PackedTensorAccessor64<int_t, 2> dist,
  int_t* out_path) {
  int_t stk[MAX_DIST * 2 + 10];
  int64_t stk_ptr = 0;
  stk[0] = i;
  stk[1] = j;
  stk[2] = max_dist;
  stk[3] = max_dist;
  while (stk_ptr >= 0) {
    const int_t stk_i = stk[2 * stk_ptr];
    const int_t stk_j = stk[2 * stk_ptr + 1];
    const int_t next_flag = stk[2 * stk_ptr + 2];

    const int_t stk_k = pred[stk_i][stk_j];
    const int_t stk_i_k_dist = dist[stk_i][stk_k];

    if (next_flag == max_dist) {
      // push first segment
      out_path[stk_i_k_dist - 1] = stk_k;

      ++stk_ptr;
      stk[2 * stk_ptr] = stk_i;
      stk[2 * stk_ptr + 1] = stk_k;
      if (stk_i_k_dist > 1) {
        stk[2 * (stk_ptr + 1)] = max_dist;
        stk[2 * (stk_ptr + 1) + 1] = max_dist;
      } else {
        --stk_ptr;
      }
    } else if (next_flag == stk_i) {
      // push second segment
      const int_t stk_k_j_dist = dist[stk_k][stk_j];
      ++stk_ptr;
      stk[2 * stk_ptr] = stk_k;
      stk[2 * stk_ptr + 1] = stk_j;
      if (stk_k_j_dist > 1) {
        stk[2 * (stk_ptr + 1)] = max_dist;
        stk[2 * (stk_ptr + 1) + 1] = max_dist;
      } else {
        --stk_ptr;
      }
      out_path += stk_i_k_dist;
    } else if (next_flag == stk_k) {
      // callback
      out_path -= stk_i_k_dist;
      --stk_ptr;
    }
  }
}

template <typename int_t, size_t MAX_DIST>
__global__ void gen_edge_input_kernel(
  const int_t num_nodes,
  const int_t max_dist,
  const torch::PackedTensorAccessor64<int_t, 2> pred,
  const torch::PackedTensorAccessor64<int_t, 2> dist,
  const int_t num_edge_features,
  const torch::PackedTensorAccessor64<int_t, 3> edge_features,
  torch::PackedTensorAccessor64<int_t, 4> output_edge_features) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int j = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes) {
    int_t path[MAX_DIST];
    const int_t i_j_dist = dist[i][j];
    if (i_j_dist > 1 && i_j_dist != max_dist) {
      get_path<int_t, MAX_DIST>(num_nodes, max_dist, i, j, pred, dist, path);
    }
    if (i_j_dist != max_dist) {
      int start = i;
      for (int e = 0; e < i_j_dist; ++e) {
        const int end = (e == i_j_dist - 1 ? j : path[e]);
        for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
          output_edge_features[i][j][e][feature_index] = edge_features[start][end][feature_index];
        }
        start = end;
      }
    }
  }
}

void gen_edge_input_cuda(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  const size_t num_nodes = pred.size(0);
  const size_t grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const size_t grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  AT_DISPATCH_INTEGRAL_TYPES(pred.type(), "gen_edge_input_kernel", ([&] {
    auto dist_accessor = dist.packed_accessor64<scalar_t, 2>();
    auto pred_accessor = pred.packed_accessor64<scalar_t, 2>();
    auto edge_features_accessor = edge_features.packed_accessor64<scalar_t, 3>();
    auto output_edge_features_accessor = output_edge_features.packed_accessor64<scalar_t, 4>();
    if (max_dist <= 16) {
      gen_edge_input_kernel<scalar_t, 16><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 32) {
      gen_edge_input_kernel<scalar_t, 32><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 64) {
      gen_edge_input_kernel<scalar_t, 64><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 128) {
      gen_edge_input_kernel<scalar_t, 128><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 256) {
      gen_edge_input_kernel<scalar_t, 256><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 512) {
      gen_edge_input_kernel<scalar_t, 512><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else {
      std::cout << "error, max_dist = " << max_dist << " is not supported by gen_edge_input." << std::endl;
      exit(-1);
    }
  }));
}
