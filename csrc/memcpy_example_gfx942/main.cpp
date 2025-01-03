#include <stdio.h>
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

#define HSACO "memcpy_tilek4_matrix_kernel_gfx942.hsaco"
#define HSA_KERNEL "memcpy_tilek4_matrix_kernel_gfx942"


#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

template<typename T>
void rand_vec(T * seq, size_t len){
    static std::random_device rd;   // seed
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<T> dist(-10.0, 10.0);

    for(size_t i=0;i<len;i++) seq[i] = dist(mt);
}

static inline bool valid_vector( const float* ref, const float* pred, int n, double nrms = 1e-6 )
{
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    for( int i=0; i<n; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>3e-5){
#ifdef ASSERT_ON_FAIL
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n",i,ri,pi,((uint32_t*)pred)[i],delta);
#endif
            pp_err++;
        }
#endif
    }
    //printf("nrms:%lf, s0:%lf, s1:%lf\n",sqrt(s0/s1),s0,s1);
    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int num_cu;
    char* gcn_arch;
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        num_cu = dev_prop.multiProcessorCount;
        gcn_arch = dev_prop.gcnArchName;
        //if(gcn_arch >= 1000)
        //    num_cu *= 2;
    }
    
    num_cu = 80;
    int thread_block_size = 256;

    int total_loop=4;
    int warm_ups = 2;
    int i;
    int bdx = thread_block_size;
    int gdx = num_cu;
    float * host_in, * host_out, *dev_in, *dev_out;
    int loops_per_block = 256;
    int total_floats = 1024 * 320;

    host_in   = new float[total_floats];
    host_out  = new float[total_floats];
    HIP_CALL(hipMalloc(&dev_in,  sizeof(float) * total_floats));
    HIP_CALL(hipMalloc(&dev_out, sizeof(float) * total_floats));

    rand_vec(host_in, total_floats);

    HIP_CALL(hipMemcpy(dev_in, host_in, sizeof(float)*total_floats, hipMemcpyHostToDevice));

    printf("memcpy, input:%p, output:%p, floats:%d\n",dev_in,dev_out, total_floats);

    struct __attribute__((packed)){
        float * input;
        float * output;
        int     loops_per_block;
        int     cu_number;
    } args;
    size_t arg_size = sizeof(args);
    args.input = dev_in;
    args.output = dev_out;
    args.loops_per_block = 1;
    args.cu_number = 320;

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};

    for(i=0;i<warm_ups;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);

    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    HIP_CALL(hipMemcpy(host_out, dev_out, sizeof(float)*total_floats, hipMemcpyDeviceToHost));

    bool is_valid = valid_vector(host_in, host_out, total_floats);
    if(!is_valid){
        printf("not valid, please check\n");
    }

    float time_per_loop_ms = elapsed_ms/total_loop;
    float gbps = total_floats*2*sizeof(float) / time_per_loop_ms / 1000 / 1000;
    printf("gbps:%f\n",gbps);

    delete [] host_in;
    delete [] host_out;
    hipFree(dev_in);
    hipFree(dev_out);
}
