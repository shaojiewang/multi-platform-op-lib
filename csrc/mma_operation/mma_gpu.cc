#include <stdio.h>
#include <iostream>
#include <random>

#include "mma_rocm.hpp"

int main(int argc, char** argv)
{
    int num_cu;
    hipDeviceProp_t device_prop;
    hipEvent_t event_start, event_end;
    hipDevice_t device;
    HIP_CALL(hipGetDevice(&device));
    HIP_CALL(hipGetDeviceProperties(&device_prop, device));
    num_cu = device_prop.multiProcessorCount;

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
  
    void* ptr_in;
    HIP_CALL(hipMalloc(&ptr_in, gdx * bdx * sizeof(bf16x4_t) * inst_iter * (1 + total_loop))); 
     
    for(i = 0; i < warm_ups; i++)
    {
        mma_launcher(ptr_in, random_seed, inst_iter, gdx, bdx);
    }
    HIP_CALL(hipEventCreate(&event_start));
    HIP_CALL(hipEventCreate(&event_end));

    (hipCtxSynchronize());
    HIP_CALL(hipEventRecord(event_start, NULL));
    for(i = 0; i < total_loop; i++)
    {
        char* tmp_ptr_in = reinterpret_cast<char*>(ptr_in) + gdx * bdx * sizeof(bf16x8_t) * (i + 1);
        mma_launcher(reinterpret_cast<void*>(tmp_ptr_in), random_seed, inst_iter, gdx, bdx);
    }
    float elapsed_ms;
    HIP_CALL(hipEventRecord(event_end, NULL));
    HIP_CALL(hipEventSynchronize(event_end));
    (hipCtxSynchronize());
    HIP_CALL(hipEventElapsedTime(&elapsed_ms, event_start, event_end));
    HIP_CALL(hipEventDestroy(event_start));
    HIP_CALL(hipEventDestroy(event_end));

    float time_per_loop = elapsed_ms / total_loop;
    //float tips = (double)inst_loop*inst_blocks*num_cu*bdx/time_per_loop/1e9;
    //argv 2~5 = M, N, K, blocks
    int MHZ = std::stoull(std::string(argv[7]));
    float SCLK = (float)MHZ / 1000.0;

    double Tflops = (double)2 * M * N * K * blocks * 4 * num_cu * (32 * inst_iter) / time_per_loop / 1e9;
    double Gflop = (double)2 * M * N * K * blocks * 4 * num_cu * (32 * inst_iter)  / 1e9;
    double TheTflops = (double)2 * M * N * K * blocks * 4 * num_cu * SCLK / cycles / 1e3;
    float RelPerf = Tflops / TheTflops;

    printf("%d, %-32s, %i, %.1f, %.3fms, %.2f, %.1f, %.3f \n", num_cu, argv[1], MHZ, Tflops, time_per_loop, Gflop, TheTflops, RelPerf);

}

