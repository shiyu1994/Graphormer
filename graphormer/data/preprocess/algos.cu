#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>

#define FLOYD_WARSHALL_BLOCK_DIM_X (16)
#define FLOYD_WARSHALL_BLOCK_DIM_Y (16)

#define CUDASUCCESS_OR_FATAL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cout << "[CUDA] " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort) exit(code);
  }
}

template <int MAX_DIST>
__global__ void floyd_warshall_set_init(
  const int num_nodes,
  const int* adjacent,
  int* output_dist) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int j = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  const int pos = i * num_nodes + j;
  if (i < num_nodes && j < num_nodes) {
    if (i == j) {
      output_dist[pos] = 0;
    } else {
      if (adjacent[pos]) {
        output_dist[pos] = 1;
      } else {
        output_dist[pos] = MAX_DIST;
      }
    }
  }
}

__global__ void floyd_warshall_one_iter_kernel(
  const int num_nodes,
  const int k,
  int* output_dist,
  int* output_pred) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int j = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  const int i_j_pos = i * num_nodes + j;
  const int i_k_pos = i * num_nodes + k;
  const int k_j_pos = k * num_nodes + j;
  if (i < num_nodes && j < num_nodes && i != k && j != k && i != j) {
    const int new_dist = output_dist[i_k_pos] + output_dist[k_j_pos];
    const int old_dist = output_dist[i_j_pos];
    if (new_dist < old_dist) {
      output_dist[i_j_pos] = new_dist;
      output_pred[i_j_pos] = k;
    }
  }
}

void floyd_warshall(
  const int num_nodes,
  const int max_dist,
  const int* adjacent,
  int* output_dist,
  int* output_pred) {
  const int grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const int grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  if (max_dist == 201) {
    floyd_warshall_set_init<201><<<grid_dim, block_dim>>>(num_nodes, adjacent, output_dist);
  } else {
    std::cout << "error, max_dist = " << max_dist << " is not supported by floyd_warshall." << std::endl;
    exit(-1);
  }
  for (int k = 0; k < num_nodes; ++k) {
    floyd_warshall_one_iter_kernel<<<grid_dim, block_dim>>>(num_nodes, k, output_dist, output_pred);
  }
}

void floyd_warshall_host(
  const int num_nodes,
  const int max_dist,
  const int* adj,
  int* output_dist,
  int* output_pred) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      const int i_j_pos = i * num_nodes + j;
      if (i == j) {
        output_dist[i_j_pos] = 0;
      } else {
        if (adj[i_j_pos]) {
          output_dist[i_j_pos] = 1;
        } else {
          output_dist[i_j_pos] = max_dist;
        }
      }
    }
  }

  for (int k = 0; k < num_nodes; ++k) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_nodes; ++i) {
      const int i_k_pos = i * num_nodes + k;
      for (int j = 0; j < num_nodes; ++j) {
        if (i == j || i == k || j == k) continue;
        const int k_j_pos = k * num_nodes + j;
        const int i_j_pos = i * num_nodes + j;
        const int new_dist = output_dist[i_k_pos] + output_dist[k_j_pos];
        const int old_dist = output_dist[i_j_pos];
        if (new_dist < old_dist) {
          output_dist[i_j_pos] = new_dist;
          output_pred[i_j_pos] = k;
        }
      }
    }
  }
}

template <int MAX_DIST>
__device__ void get_path(
  const int num_nodes,
  const int i,
  const int j,
  const int* pred,
  const int* dist,
  int* out_path) {
  int stk[MAX_DIST * 20];
  int stk_ptr = 0;
  stk[0] = i;
  stk[1] = j;
  stk[2] = -1;
  stk[3] = -1;
  while (stk_ptr >= 0) {
    const int stk_i = stk[2 * stk_ptr];
    const int stk_j = stk[2 * stk_ptr + 1];
    const int next_flag = stk[2 * stk_ptr + 2];

    const int stk_i_j_pos = stk_i * num_nodes + stk_j;
    const int stk_k = pred[stk_i_j_pos];

    const int stk_i_k_pos = stk_i * num_nodes + stk_k;
    const int stk_i_k_dist = dist[stk_i_k_pos];

    if (next_flag == -1) {
      // push first segment
      out_path[stk_i_k_dist - 1] = stk_k;

      ++stk_ptr;
      stk[2 * stk_ptr] = stk_i;
      stk[2 * stk_ptr + 1] = stk_k;
      if (stk_i_k_dist > 1) {
        stk[2 * (stk_ptr + 1)] = -1;
        stk[2 * (stk_ptr + 1) + 1] = -1;
      } else {
        --stk_ptr;
      }
    } else if (next_flag == stk_i) {
      // push second segment
      const int stk_k_j_pos = stk_k * num_nodes + stk_j;
      const int stk_k_j_dist = dist[stk_k_j_pos];
      ++stk_ptr;
      stk[2 * stk_ptr] = stk_k;
      stk[2 * stk_ptr + 1] = stk_j;
      if (stk_k_j_dist > 1) {
        stk[2 * (stk_ptr + 1)] = -1;
        stk[2 * (stk_ptr + 1) + 1] = -1;
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

template <int MAX_DIST>
__global__ void gen_edge_input_kernel(
  const int num_nodes,
  const int max_dist,
  const int* pred,
  const int* dist,
  const int num_edge_features,
  const int* edge_features,
  int* output_edge_features) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int j = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes) {
    int path[MAX_DIST];
    const int i_j_pos = i * num_nodes + j;
    const int i_j_dist = dist[i_j_pos];
    if (i_j_dist > 1 && i_j_dist != max_dist) {
      get_path<MAX_DIST>(num_nodes, i, j, pred, dist, path);
    }
    const int edge_dist_offset = i_j_pos * max_dist;
    int start = i;
    for (int e = 0; e < i_j_dist; ++e) {
      const int end = (e == i_j_dist - 1 ? j : path[e]);
      const int edge_offset = (start * num_nodes + end) * num_edge_features;
      const int feature_offset = (edge_dist_offset + e) * num_edge_features;
      for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
        output_edge_features[feature_offset + feature_index] = edge_features[edge_offset + feature_index];
      }
      start = end;
    }
  }
}

void gen_edge_input(
  const int num_nodes,
  const int max_dist,
  const int* pred,
  const int* dist,
  const int num_edge_features,
  const int* edge_features,
  int* output_edge_features) {
  const int grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const int grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  if (max_dist <= 16) {
    gen_edge_input_kernel<16><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else if (max_dist <= 32) {
    gen_edge_input_kernel<32><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else if (max_dist <= 64) {
    gen_edge_input_kernel<64><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else if (max_dist <= 128) {
    gen_edge_input_kernel<128><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else if (max_dist <= 256) {
    gen_edge_input_kernel<256><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else if (max_dist <= 512) {
    gen_edge_input_kernel<512><<<grid_dim, block_dim>>>(num_nodes, max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
  } else {
    std::cout << "error, max_dist = " << max_dist << " is not supported by gen_edge_input." << std::endl;
    exit(-1);
  }
}

void get_path_host(
  const int num_nodes,
  const int i,
  const int j,
  const int* pred,
  const int* dist,
  int* out_path) {
  const int i_j_pos = i * num_nodes + j;
  const int k = pred[i_j_pos];
  const int i_k_pos = i * num_nodes + k;
  const int i_k_dist = dist[i_k_pos];
  out_path[i_k_dist - 1] = k;
  if (i_k_dist > 1) {
    get_path_host(num_nodes, i, k, pred, dist, out_path);
  }
  const int k_j_pos = k * num_nodes + j;
  const int k_j_dist = dist[k_j_pos];
  if (k_j_dist > 1) {
    get_path_host(num_nodes, k, j, pred, dist, out_path + i_k_dist);
  }
}

void gen_edge_input_host(
  const int num_nodes,
  const int max_dist,
  const int* pred,
  const int* dist,
  const int num_edge_features,
  const int* edge_features,
  int* output_edge_features) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      const int i_j_pos = i * num_nodes + j;
      const int i_j_dist = dist[i_j_pos];
      if (i_j_dist > 1 && i_j_dist != max_dist) {
        std::vector<int> path(i_j_dist - 1);
        get_path_host(num_nodes, i, j, pred, dist, path.data());
        const int edge_dist_offset = (i * num_nodes + j) * max_dist;
        int start = i;
        for (int e = 0; e < i_j_dist; ++e) {
          const int end = (e == i_j_dist - 1 ? j : path[e]);
          const int edge_offset = (start * num_nodes + end) * num_edge_features;
          const int feature_offset = (edge_dist_offset + e) * num_edge_features;
          for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
            output_edge_features[feature_offset + feature_index] = edge_features[edge_offset + feature_index];
          }
          start = end;
        }
      }
    }
  }
}

void test() {
  const int num_nodes = 200;
  const double edge_prob = 0.02;
  const int num_edge_features = 10;
  const int max_edge_feature_values = 200;
  const int num_threads = 16;
  const int max_dist = num_nodes + 1;
  std::uniform_real_distribution<double> rand_dist;
  std::vector<std::mt19937> rand_eng(num_threads);
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    rand_eng[thread_index] = std::mt19937(thread_index);
  }
  std::vector<int> adj(num_nodes * num_nodes, 0);
  const int block_size = (num_nodes + num_threads - 1) / num_threads;
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    const int start = thread_index * block_size;
    const int end = std::min(start + block_size, num_nodes);
    for (int i = start; i < end; ++i) {
      for (int j = 0; j < num_nodes; ++j) {
        const int pos = i * num_nodes + j;
        if (i != j) {
          const double prob = rand_dist(rand_eng[thread_index]);
          if (prob <= edge_prob) {
            adj[pos] = 1;
          } else {
            adj[pos] = 0;
          }
        }
      }
    }
  }

  std::vector<int> dist(num_nodes * num_nodes, 0);
  std::vector<int> pred(num_nodes * num_nodes, 0);
  int* cuda_adj = nullptr;
  int* cuda_dist = nullptr;
  int* cuda_pred = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_dist, dist.size() * sizeof(int)));
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_pred, pred.size() * sizeof(int)));
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_adj, adj.size() * sizeof(int)));
  CUDASUCCESS_OR_FATAL(cudaMemcpy(cuda_adj, adj.data(), adj.size() * sizeof(int), cudaMemcpyHostToDevice));
  const auto cuda_start = std::chrono::steady_clock::now();
  floyd_warshall(num_nodes, max_dist, cuda_adj, cuda_dist, cuda_pred);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());
  const auto cuda_end = std::chrono::steady_clock::now();
  const double cuda_cost = (static_cast<std::chrono::duration<double>>(cuda_end - cuda_start)).count();
  std::cout << "cuda cost time " << cuda_cost << std::endl;
  CUDASUCCESS_OR_FATAL(cudaMemcpy(dist.data(), cuda_dist, dist.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CUDASUCCESS_OR_FATAL(cudaMemcpy(pred.data(), cuda_pred, pred.size() * sizeof(int), cudaMemcpyDeviceToHost));

  std::vector<int> host_dist(num_nodes * num_nodes, 0);
  std::vector<int> host_pred(num_nodes * num_nodes, 0);
  const auto host_start = std::chrono::steady_clock::now();
  floyd_warshall_host(num_nodes, max_dist, adj.data(), host_dist.data(), host_pred.data());
  const auto host_end = std::chrono::steady_clock::now();
  const double host_cost = (static_cast<std::chrono::duration<double>>(host_end - host_start)).count();
  std::cout << "host cost time " << host_cost << std::endl;
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      const int i_j_pos = i * num_nodes + j;
      if (host_dist[i_j_pos] != dist[i_j_pos]) {
        std::cout << "error in dist " << i << " " << j << " " << host_dist[i_j_pos] << " " << dist[i_j_pos] << std::endl;
      }
      if (host_pred[i_j_pos] != pred[i_j_pos]) {
        std::cout << "error in pred " << i << " " << j << " " << host_pred[i_j_pos] << " " << pred[i_j_pos] << std::endl;
      }
    }
  }
  std::cout << "tests shortest path finished" << std::endl;

  std::vector<int> host_edge_features(num_nodes * num_nodes * num_edge_features);
  std::vector<int> host_output_edge_features(num_nodes * num_nodes * max_dist * num_edge_features);
  std::vector<double> edge_feature_value_prob(max_edge_feature_values, 1.0 / max_edge_feature_values);
  std::discrete_distribution<int> edge_feature_dist(edge_feature_value_prob.begin(), edge_feature_value_prob.end());

  std::cout << "genearting edge features on host" << std::endl;
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    const int start = thread_index * block_size;
    const int end = std::min(start + block_size, num_nodes);
    for (int i = start; i < end; ++i) {
      for (int j = 0; j < num_nodes; ++j) {
        const int i_j_pos = i * num_nodes + j;
        const int offset = i_j_pos * num_edge_features;
        for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
          host_edge_features[offset + feature_index] = edge_feature_dist(rand_eng[thread_index]);
        }
      }
    }
  }

  std::cout << "genearting edge input on host" << std::endl;
  const auto host_edge_input_start = std::chrono::steady_clock::now();
  gen_edge_input_host(num_nodes, max_dist, pred.data(), dist.data(), num_edge_features, host_edge_features.data(), host_output_edge_features.data());
  const auto host_edge_input_end = std::chrono::steady_clock::now();
  const double host_edge_input_cost = (static_cast<std::chrono::duration<double>>(host_edge_input_end - host_edge_input_start)).count();
  std::cout << "host edge input cost time " << host_edge_input_cost << std::endl;

  std::cout << "allocating cuda memory" << std::endl;
  int* cuda_edge_features = nullptr;
  cudaMalloc(&cuda_edge_features, sizeof(int) * host_edge_features.size());
  cudaMemcpy(cuda_edge_features, host_edge_features.data(), sizeof(int) * host_edge_features.size(), cudaMemcpyHostToDevice);
  int* cuda_output_edge_features = nullptr;
  cudaMalloc(&cuda_output_edge_features, sizeof(int) * host_output_edge_features.size());

  std::cout << "generating edge input on cuda" << std::endl;
  const auto cuda_edge_input_start = std::chrono::steady_clock::now();
  gen_edge_input(num_nodes, max_dist, cuda_pred, cuda_dist, num_edge_features, cuda_edge_features, cuda_output_edge_features);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());
  const auto cuda_edge_input_end = std::chrono::steady_clock::now();
  const double cuda_edge_input_cost = (static_cast<std::chrono::duration<double>>(cuda_edge_input_end - cuda_edge_input_start)).count();
  std::cout << "cuda edge input cost time " << cuda_edge_input_cost << std::endl;
  std::vector<int> output_edge_features(host_output_edge_features.size());
  cudaMemcpy(output_edge_features.data(), cuda_output_edge_features, sizeof(int) * host_output_edge_features.size(), cudaMemcpyDeviceToHost);

  /*std::cout << "comparing path results" << std::endl;
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      const int i_j_pos = i * num_nodes + j;
      const int i_j_dist = host_dist[i_j_pos];
      if (i_j_dist != max_dist) {
        for (int d = 0; d < i_j_dist - 1; ++d) {
          const int host_result = host_paths[i_j_pos][d];
          const int cuda_result = cuda_to_host_paths[i_j_pos * max_dist + d];
          if (host_result != cuda_result) {
            std::cout << "path " << i << " " << j << " " << d << " " << host_result << " vs " << cuda_result << std::endl;
            exit(-1);
          }
        }
      }
    }
  }*/

  std::cout << "comparing edge input results" << std::endl;
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      const int i_j_pos = i * num_nodes + j;
      const int i_j_dist = host_dist[i_j_pos];
      if (i_j_dist != max_dist) {
        for (int d = 0; d < i_j_dist - 1; ++d) {
          const int offset = (i_j_pos * max_dist + d) * num_edge_features;
          for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
            const int cuda_result = output_edge_features[offset + feature_index];
            const int host_result = host_output_edge_features[offset + feature_index];
            if (cuda_result != host_result) {
              std::cout << "output edge feature" << " " << i << " " << j << " " << d << " " << feature_index << " " << host_result << " vs " << cuda_result << std::endl;
              exit(-1);
            }
          }
        }
      }
    }
  }
}

int main() {
  test();
  return 0;
}
