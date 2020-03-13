
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#define ZERO_THRESHOLD (1e-5)
int debug = 0;
#ifdef _GPU_
#include <cuda_runtime.h>
#define WARP_SIZE (32)
#define COMP_BLOCK_SIZE (512)
#define MAX_WARP_COUNT (32)

__inline__ __device__ float warp_reduce(float val) {
#pragma unroll
  for (int i = WARP_SIZE >> 1; i > 0; i >>= 1)
    val += __shfl_down_sync(0xffffffff, val, i);
  return val;
}

__global__ void gram_schmidt_scale(int m, int n, float* mat, int k, int* ret) {

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int lane_idx   = thread_idx % WARP_SIZE;

  __shared__ float r;

  if (lane_idx == 0) r = 0;

  float a[WARP_SIZE];

  int cnt = 0;
  for (int w = 0; w < m; w += WARP_SIZE) {
    int i = w + lane_idx;

    float val = 0;
    if (i < m) {
      val      = mat[i * n + k];
      a[cnt++] = val;
    }

    val = warp_reduce(val * val);
    if (lane_idx == 0) r += val;
  }

  if (lane_idx == 0) r = sqrt(r);
  __syncwarp();

  int is_zero = r < ZERO_THRESHOLD;

  cnt = 0;
  for (int w = 0; w < m; w += WARP_SIZE) {
    int i = w + lane_idx;

    if (i >= m)
      continue;
    else if (is_zero)
      mat[i * n + k] = 0;
    else
      mat[i * n + k] = a[cnt++] / r;
  }

  if (ret)
    *ret = is_zero;
}

__global__ void gram_schmidt_combine(int m, int n, float* mat, int k, int* is_zero) {

  if (is_zero && *is_zero) return;

  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_idx   = thread_idx / WARP_SIZE;
  int lane_idx   = thread_idx % WARP_SIZE;

  int j = warp_idx + k + 1;
  if (j >= n) return;

  int warp_blk_idx = threadIdx.x / WARP_SIZE;

  __shared__ float s_r[MAX_WARP_COUNT];
  __shared__ float s_q[MAX_WARP_COUNT * WARP_SIZE];

  if (lane_idx == 0) s_r[warp_blk_idx] = 0;

  if (thread_idx < m)
    s_q[thread_idx] = mat[thread_idx * n + k];
  __syncthreads();

  float a[WARP_SIZE];

  int cnt = 0;
  for (int w = 0; w < m; w += WARP_SIZE) {
    int i = w + lane_idx;

    float val = 0;
    if (i < m) {
      val      = mat[i * n + j];
      a[cnt++] = val;
      val *= s_q[i];
    }

    val = warp_reduce(val);
    if (lane_idx == 0) s_r[warp_blk_idx] += val;
  }

  __syncwarp();

  auto r = s_r[warp_blk_idx];

  if (r == 0)
    return;

  cnt = 0;
  for (int w = 0; w < m; w += WARP_SIZE) {
    int i = w + lane_idx;

    if (i < m)
      mat[i * n + j] = a[cnt++] - r * s_q[i];
  }
}
#endif

void gram_schmidt_gpu(int m, int n, float* mat) {

  int* is_zero;
  cudaMalloc((void**)&is_zero, sizeof(int));

  for (int k = 0; k < n; ++k) {
    gram_schmidt_scale<<<1, WARP_SIZE>>>(m, n, mat, k, is_zero);
    gram_schmidt_combine<<<(n - k) * WARP_SIZE / COMP_BLOCK_SIZE + 1, COMP_BLOCK_SIZE>>>(m, n, mat, k, is_zero);
  }
}

void gram_schmidt_cpu(int m, int n, float* mat) {

  for (int k = 0; k < n; ++k) {
    float r = 0;
    for (int i = 0; i < m; i++)
      r = r + mat[i * n + k] * mat[i * n + k];
    r = sqrt(r);

    if (debug) cout << "r" << k << k << ": " << r << endl;

    for (int i = 0; i < m; i++) {
      if (r < ZERO_THRESHOLD)
        mat[i * n + k] = 0;
      else
        mat[i * n + k] = mat[i * n + k] / r;


      if (debug) cout << "q" << i << k << ": " << mat[i * n + k] << " ";
    }

    if (debug) cout << endl;

    for (int j = k + 1; j < n; ++j) {
      float r = 0;
      for (int i = 0; i < m; i++) r += mat[i * n + k] * mat[i * n + j];
      for (int i = 0; i < m; i++) {
        mat[i * n + j] = mat[i * n + j] - r * mat[i * n + k];
      }
      if (debug) {
        printf("r%d%d=%f ", k, j, r);
        for (int i = 0; i < m; i++) cout << "a" << j << ": " << mat[i * n + j] << " ";
        cout << endl;
      }
    }

    if (debug) printf("\n-------------------\n");
  }
}

int main(int argc, char* argv[]) {

  int m, n;

  /* let user set the dimension of matrix A */
  std::cout << "Enter the vector length ";
  std::cin >> m;
  std::cout << "Enter the # of vectors ";
  std::cin >> n;

  assert(m > 0 && n > 0);

  int debug = m <= 10 && n <= 10;

  float* b = NULL;
  if (m == 3 && n == 4) {
    static float x[12] = {
      1, 2, 0, 1,
      1, 2, 0, 2,
      1, 3, 1, 3};

    b = (float*)x;

    //known solution
    // 5.773503e-01 -4.082482e-01 0.000000e+00 -7.071067e-01
    // 5.773503e-01 -4.082482e-01 0.000000e+00  7.071068e-01
    // 5.773503e-01  8.164967e-01 0.000000e+00  0.000000e+00
  } else if (m == 4 && n == 3) {
    static float x[12] = {
      1, 1, 1,
      2, 2, 0,
      3, 0, 0,
      0, 0, 1};

    b = (float*)x;

    //known solution
    // 2.672612e-01  3.585686e-01  5.962848e-01
    // 5.345225e-01  7.171372e-01 -2.981424e-01
    // 8.017837e-01 -5.976143e-01  0.000000e+00
    // 0.000000e+00  0.000000e+00  7.453560e-01
  } else if (m == 3 && n == 3) {
    static float x[9] = {
      1, 2, 1,
      0, 1, 2,
      1, 2, 0};

    b = (float*)x;

    //known solution
    // 7.071068e-01 1.192093e-07  7.071065e-01
    // 0.000000e+00 1.000000e+00  0.000000e+00
    // 7.071068e-01 1.192093e-07 -7.071071e-01
  } else {
    b = new float[m * n];

    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        b[i * n + j] = rand() % 10;

    if (m <= 10 && n <= 10) {
      printf("\n\nimport numpy as np\nA = np.array([");
      for (int i = 0; i < m; ++i) {
        printf("[");
        for (int j = 0; j < n; ++j) {
          printf("%e%s", b[i * n + j], j < n - 1 ? "," : "");
        }
        printf("]%s", i < m - 1 ? "," : "");
      }
      printf("])\n");
      printf("q, r = np.linalg.qr(A)\nprint q\n\n");
    }
  }
#ifdef _GPU_
  float* dev_b;
  cudaMalloc((void**)&dev_b, sizeof(float) * m * n);
  cudaMemcpy(dev_b, b, sizeof(float) * m * n, cudaMemcpyHostToDevice);
#endif
  //CPU
  {
    printf("==========CPU start==========\n");
    auto start = std::chrono::high_resolution_clock::now();
    gram_schmidt_cpu(m, n, b);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " usec" << std::endl;
  }
#ifdef _GPU_
  //GPU
  {
    printf("==========GPU start==========\n");
    assert(m < 1024);
    auto start = std::chrono::high_resolution_clock::now();
    gram_schmidt_gpu(m, n, dev_b);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " usec" << std::endl;
  }

  float* sol_b = new float[m * n];
  cudaMemcpy(sol_b, dev_b, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  //compare CPU vs GPU
  int num_mismatch = 0;
  for (int i = 0; i < m * n; ++i) {
    int mismatch = fabs(sol_b[i] - b[i]) > ZERO_THRESHOLD;

    num_mismatch += mismatch;
    if (mismatch && num_mismatch++ < 10)
      printf("mistmatch at %d (%e vs %e)\n", i, sol_b[i], b[i]);
  }
#endif

  printf("Numerical verification for orthonormality checking...\n");
  int num_err = 0;
  for (int k = 0; k < n; ++k) {
    for (int j = k + 1; j < n; ++j) {
      float r = 0;
      for (int i = 0; i < m; i++) r += b[i * n + k] * b[i * n + j];

      int err = fabs(r) > ZERO_THRESHOLD;

      num_err += err;
      if (err && num_err++ < 10)
        printf("(%d,%d)=%e\n", k, j, r);
    }
  }

  printf("...completed with %d errors\n", num_err);

  if (debug) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j)
        printf("%e ", b[i * n + j]);
      printf("\n");
    }
  }

  return 0;
}
