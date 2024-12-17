#include <stdio.h>
#include <string>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                      \
} while(0)

#define HSACO "kernel.co"
#define HSA_KERNEL "kernel_func"

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
    }

    int total_loop= 100;
    int warm_ups = 5;
    int i;
    int bdx = 256;
    int gdx = num_cu;
    
    int num_inst = std::stoull(std::string(argv[1]));
    float data_in_bytes = (float)(num_inst) * 16.0 * bdx * num_cu * 16;
    srand(time(NULL));
    float rand_seed = ((float)(rand() % 1000))/1000.0;
    struct {
        float rand_seed;
        unsigned int inst_iter;
        int s_nop;
    } args;
    size_t arg_size = sizeof(args);
    args.inst_iter = num_inst;
    args.rand_seed = rand_seed;

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipCtxSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipCtxSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms / total_loop;
    int MHZ = std::stoull(std::string(argv[2]));
    float SCLK = (float)MHZ / 1000.0;

    float avg_latency = time_per_loop / num_inst;
    float lds_bw = data_in_bytes / 1000.0 / 1000.0 / 1000.0 / time_per_loop;

    printf("num_inst=%d, bytes=%f, time=%f, avg_latency=%f, avg_bw=%f\n", num_inst, data_in_bytes, time_per_loop, avg_latency, lds_bw);
}

