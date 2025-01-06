#include <stdio.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iostream>
#include <vector>
#include <array>

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s\n",(int)err,#call);    \
        exit(0);            \
    }                      \
} while(0)

#define ARRAY_MEMCPY_HSACO "memcpy_x4_kernel_gfx942.hsaco"
#define HSA_ARRAY_MEMCPY_KERNEL "memcpy_x4_kernel_gfx942"

#define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

#define ERROR_LOG(msg) do{          \
    std::cerr << msg << std::endl;  \
    exit(1);                        \
} while(0)

const static std::array<std::string, 2> MATRIX_MEMCPY_KERNEL_LIST = {
    "memcpy_matrix_loop_by_col_kernel_gfx942",
    "memcpy_matrix_loop_by_row_kernel_gfx942"
};

template<typename T>
void rand_tensor(T * seq, size_t len){
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
                //printf("diff at %4d, ref:%lf, pred:%lf(0x%08x), d:%lf\n",i,ri,pi,((uint32_t*)pred)[i],delta);
        if(delta>0){
#ifdef ASSERT_ON_FAIL
            if(1)
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

template<typename DataType>
struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;
    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(&p_mem_, mem_size * sizeof(DataType));
    }

    DataType* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    DataType* p_mem_;
};

struct LaunchParams {
    int num_cu;
    int gdx, bdx;
    int total_loop;
    int warm_ups;
};

struct ArrayMemcpyLaunchParams: LaunchParams {
    int loops_per_block;
    int workload;
};

struct MatrixMemcpyLaunchParams: LaunchParams {
    int M, K;
    // 1: loop by col; 2: loop by row
    int direction;
    int blk_tile_k;
    int workload;
};

template<typename DataType>
struct __attribute__((packed)) MemcpyKernelParams {
    DataType* input;
    DataType* output;
    int loops_per_block;
    int num_cu;
};

template<typename DataType>
struct __attribute__((packed)) MatrixMemcpyKernelParams {
    DataType* input;
    DataType* output;
    int loops_per_block;
    int K;
    int blk_tile_k;
    int log_blk_tile_k;
    int block_stride;
};

template<typename DataType>
class GFX942Test {
public:
    static void test_array_memcpy(const ArrayMemcpyLaunchParams& launch_params) {
        hipModule_t module;
        hipFunction_t kernel_func;
        HIP_CALL(hipSetDevice(0));
        HIP_CALL(hipModuleLoad(&module, ARRAY_MEMCPY_HSACO));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_ARRAY_MEMCPY_KERNEL));

        const int total_len = launch_params.num_cu * launch_params.bdx * launch_params.workload * launch_params.loops_per_block;
        SimpleDeviceMem<DataType> dev_in(total_len), dev_out(total_len);

        MemcpyKernelParams<DataType> kernel_params{dev_in.GetDeviceBuffer(), dev_out.GetDeviceBuffer(), launch_params.loops_per_block, launch_params.num_cu};
        float elapsed_ms = time(kernel_func, launch_params, kernel_params, total_len);

        float time_per_loop_ms = elapsed_ms / launch_params.total_loop;
        float gbps = total_len*2*sizeof(DataType) / time_per_loop_ms / 1000 / 1000;
        printf("gbps:%f\n",gbps);
    }

    static void test_matrix_memcpy(const MatrixMemcpyLaunchParams& launch_params) {
        hipModule_t module;
        hipFunction_t kernel_func;
        HIP_CALL(hipSetDevice(0));
        std::string kernel_name = MATRIX_MEMCPY_KERNEL_LIST[launch_params.direction];
        HIP_CALL(hipModuleLoad(&module, (kernel_name + ".hsaco").c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

        int loops_per_block = launch_params.M / (launch_params.bdx * launch_params.workload * 4 / launch_params.blk_tile_k);
        const int total_len = launch_params.M * launch_params.K;
        int blk_stride = launch_params.bdx / launch_params.blk_tile_k * launch_params.workload * launch_params.K;
        int log2_blk_row_len = static_cast<int>(std::log2(launch_params.blk_tile_k / launch_params.workload));
        if (launch_params.direction == 1) {
            blk_stride = launch_params.blk_tile_k;
            loops_per_block = launch_params.K / launch_params.blk_tile_k / 4;
        }

        SimpleDeviceMem<DataType> dev_in(total_len), dev_out(total_len);
        printf("data size: %dMB, dev input addr: %p, dev output addr: %p\nM: %d, K: %d, block stride: %d,"
               " loops per block: %d\n",
                total_len*sizeof(DataType)/1024/1024, dev_in.GetDeviceBuffer(), dev_out.GetDeviceBuffer(),
                launch_params.M, launch_params.K, blk_stride, loops_per_block);
        MatrixMemcpyKernelParams<DataType> kernel_params{
            dev_in.GetDeviceBuffer(), dev_out.GetDeviceBuffer(),
            loops_per_block, launch_params.K, launch_params.blk_tile_k,
            log2_blk_row_len, blk_stride
        };
        float elapsed_ms = time(kernel_func, launch_params, kernel_params, total_len);

        float time_per_loop_ms = elapsed_ms / launch_params.total_loop;
        float gbps = total_len*2*sizeof(DataType) / time_per_loop_ms / 1000 / 1000;
        printf("gbps:%f\n",gbps);
    }

private:
    template<typename KernelParams>
    static float time(hipFunction_t kernel, const LaunchParams& launch_params, KernelParams& kernel_params, const int total_len) {
        std::vector<DataType> host_in(total_len), host_out(total_len);

        size_t arg_size = sizeof(kernel_params);
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernel_params, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};

        auto rand_device_input = [&]() {
            rand_tensor(host_in.data(), total_len);
            HIP_CALL(hipMemcpy(kernel_params.input, host_in.data(), sizeof(DataType)*total_len, hipMemcpyHostToDevice));
        };

        auto verify = [&]() {
            HIP_CALL(hipMemcpy(host_out.data(), kernel_params.output, sizeof(DataType)*total_len, hipMemcpyDeviceToHost));
            bool is_valid = valid_vector(host_in.data(), host_out.data(), total_len);
            if(!is_valid){
                printf("not valid, please check\n");
            }
        };

        for (int i = 0; i < launch_params.warm_ups; ++i) {
            rand_device_input();
            // f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams, extra
            HIP_CALL(hipModuleLaunchKernel(kernel, launch_params.gdx,1,1, launch_params.bdx,1,1,  0, 0, NULL, (void**)&config));
            verify();
        }

        hipEvent_t evt_00, evt_11;
        hipEventCreate(&evt_00);
        hipEventCreate(&evt_11);
        hipDeviceSynchronize();

        float total_time = 0.f;
        for (int i = 0; i < launch_params.total_loop; ++i) {
            rand_device_input();

            hipEventRecord(evt_00, NULL);
            HIP_CALL(hipModuleLaunchKernel(kernel, launch_params.gdx,1,1, launch_params.bdx,1,1,  0, 0, NULL, (void**)&config));
            hipEventRecord(evt_11, NULL);
            hipEventSynchronize(evt_11);
            float elapsed_ms;
            hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);

            total_time += elapsed_ms;
            verify();
        }

        hipDeviceSynchronize();

        hipEventDestroy(evt_00);
        hipEventDestroy(evt_11);

        return total_time;
    }

};

int main(int argc, char ** argv){
    if (argc <= 1) {
        ERROR_LOG("must provide args\n     1 for array: 1 $gridDim\n     2 for matrix: 2 $direction $M $tile_k $gridDim");
    }
    const unsigned test_type = atoi(argv[1]);

    switch (test_type)
    {
    case 1:
        {
            const int num_cu = atoi(argv[2]);
            constexpr int thread_block_size = 256;

            constexpr int total_loop = 4;
            constexpr int warm_ups = 2;
            constexpr int bdx = thread_block_size;  // blockDim
            const int gdx = num_cu;                 // gridDim
            constexpr int loops_per_block = 256;
            constexpr int workload = 16;            // float4 * 4
            ArrayMemcpyLaunchParams params {
                num_cu,
                gdx,
                bdx,
                total_loop,
                warm_ups,
                loops_per_block,
                workload
            };

            GFX942Test<float>::test_array_memcpy(params);
        }
        break;
    case 2:
        {
            const int direction = atoi(argv[2]);
            const int M = atoi(argv[3]);
            const int tile_k = atoi(argv[4]);
            const int num_cu = atoi(argv[5]);
            if (direction != 0 && direction != 1) {
                ERROR_LOG("direction arg only support {0: loop by col, 1: loop by row}");
            }
            if (tile_k <= 0 || (tile_k & (tile_k - 1)) != 0) {
                ERROR_LOG("tile_k must be power of 2");
            }

            constexpr int workload = 4; // float4
            constexpr int total_loop = 4;
            constexpr int warm_ups = 2;
            constexpr int bdx = 256;  // blockDim
            const int gdx = num_cu;   // gridDim

            MatrixMemcpyLaunchParams params {
                num_cu,
                gdx,
                bdx,
                total_loop,
                warm_ups,
            };
            params.direction = direction;
            params.blk_tile_k = tile_k;
            params.workload = workload;

            if (direction == 0) {
                params.K = num_cu * tile_k;
                params.M = M;
            } else if (direction == 1) {
                params.K = M;
                params.M = num_cu * (bdx * workload / tile_k);
            }


            GFX942Test<float>::test_matrix_memcpy(params);
        }
        break;
    default:
        ERROR_LOG("first arg only support {1: array, 2: matrix}");
        break;
    }

}

