#include <stdio.h>
#include <iostream>
#include <random>

#include "mma_rocm.hpp"

int main(int argc, char** argv)
{
    int num_cu;
    hipDeviceProp_t device_prop;
    hipEvent_t event_start, event_end;
    hipDevice_t devive;
    HIP_CALL(hipGetDevice(&device));
    HIP_CALL(hipGetDeviceProperties(&device_prop, device));
    num_cu = device_prop.multiProcessorCount;

    int total_loop= 100;
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
    float rand_seed = ((float)(rand() % 1000))/1000.0;
   
     
    for(i = 0; i < warm_ups; i++)
    {
        mma_launcher(nullptr, random_seed, inst_iter, gdx, bdx);
    }
    hipEventCreate(&event_start);
    hipEventCreate(&event_end);

    hipCtxSynchronize();
    hipEventRecord(event_start, NULL);
    for(i = 0; i < total_loop; i++)
    {
        mma_launcher(nullptr, random_seed, inst_iter, gdx, bdx);
    }
    float elapsed_ms;
    hipEventRecord(event_end, NULL);
    hipEventSynchronize(event_end);
    hipCtxSynchronize();
    hipEventElapsedTime(&elapsed_ms, event_start, event_end);
    hipEventDestroy(event_start);
    hipEventDestroy(event_end);

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

