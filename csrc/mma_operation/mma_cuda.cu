#include <stdio.h>
#include <iostream>
#include <random>

#include "mma_cuda.hpp"

int main(int argc, char** argv)
{
  int num_cu = get_sm_count();
  cudaEvent_t event_start, event_end;
  int total_loop= 10;
  int warm_ups = 5;
  int i;
  int bdx = 256;
  int gdx = num_cu;

  int M = std::stoull(std::string(argv[2]));
  int N = std::stoull(std::string(argv[3]));
  int K = std::stoull(std::string(argv[4]));
  int blocks = std::stoull(std::string(argv[5]));
  int cycles = std::stoull(std::string(argv[6]));
  unsigned int inst_iter = static_cast<unsigned int>(static_cast<unsigned long long>(2048)*1024*8/(M*N*K*blocks));
  srand(time(NULL));
  float random_seed = ((float)(rand() % 1000))/1000.0;
 
  printf("size if %d\n", gdx * bdx * sizeof(bf16x8_t) * inst_iter * (1 + total_loop));
 
  void* ptr_in;
  CUDA_CHECK(cudaMalloc(&ptr_in, gdx * bdx * sizeof(bf16x8_t) * inst_iter * (1 + total_loop))); 
     
  for(i = 0; i < warm_ups; i++)
  {
    mma_launcher<__nv_bfloat16, __nv_bfloat16, float>(reinterpret_cast<float*>(ptr_in), random_seed, inst_iter, gdx, bdx);
  }
  CUDA_CHECK(cudaEventCreate(&event_start));
  CUDA_CHECK(cudaEventCreate(&event_end));

  (cuCtxSynchronize());
  CUDA_CHECK(cudaEventRecord(event_start, NULL));
  for(i = 0; i < total_loop; i++)
  {
    char* tmp_ptr_in = reinterpret_cast<char*>(ptr_in);// + gdx * bdx * sizeof(bf16x8_t) * (0 + 1);
    mma_launcher<__nv_bfloat16, __nv_bfloat16, float>(reinterpret_cast<float*>(tmp_ptr_in), random_seed, inst_iter, gdx, bdx);
  }
  float elapsed_ms;
  CUDA_CHECK(cudaEventRecord(event_end, NULL));
  CUDA_CHECK(cudaEventSynchronize(event_end));
  (cuCtxSynchronize());
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, event_start, event_end));
  CUDA_CHECK(cudaEventDestroy(event_start));
  CUDA_CHECK(cudaEventDestroy(event_end));

  float time_per_loop = elapsed_ms / total_loop;
  //float tips = (double)inst_loop*inst_blocks*num_cu*bdx/time_per_loop/1e9;
  //argv 2~5 = M, N, K, blocks
  int MHZ = std::stoull(std::string(argv[7]));
  float SCLK = (float)MHZ / 1000.0;

  double Tflops = (double)2 * M * N * K * blocks * 4 * num_cu * (32 * inst_iter) / time_per_loop / 1e9;
  double Gflop = (double)2 * M * N * K * blocks * 4 * num_cu * (32 * inst_iter)  / 1e9;
  double TheTflops = 989.0; // (double)2 * M * N * K * blocks * 4 * num_cu * SCLK / cycles / 1e3;
  float RelPerf = Tflops / TheTflops;

  printf("%d, %-32s, %i, %.1f, %.3fms, %.2f, %.1f, %.3f \n", num_cu, argv[1], MHZ, Tflops, time_per_loop, Gflop, TheTflops, RelPerf);

  
}
