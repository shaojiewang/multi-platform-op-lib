import os
import torch
from torch.utils.cpp_extension import load
from functools import wraps

import torch
from functools import wraps
import numpy as np

def cuda_timer(device=None, sync=True, repetitions=1, warmup=3):
    """
    CUDA事件计时装饰器（增强版）
    
    参数:
        device (int/str): 指定使用的CUDA设备 (默认: 当前设备)
        sync (bool): 是否同步设备 (确保准确计时)
        repetitions (int): 重复测量次数 (返回平均时间)
        warmup (int): 预热次数 (不纳入计时)
    
    返回:
        tuple: (函数结果, 时间指标字典)
        时间指标包含:
            'avg_time_ms': 平均执行时间(ms)
            'min_time_ms': 最小执行时间(ms)
            'max_time_ms': 最大执行时间(ms)
            'tflops': 计算吞吐量(TFLOPS)
            'hbm_bandwidth': 内存带宽(GB/s)
            'times_ms': 所有执行时间列表(ms)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 确定使用的设备
            target_device = device if device is not None else torch.cuda.current_device()
            device_obj = torch.device(f'cuda:{target_device}')
            
            # 创建CUDA事件
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions + warmup)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions + warmup)]
            
            # 预热运行
            for i in range(warmup):
                with torch.cuda.device(device_obj):
                    result = func(*args, **kwargs)  # 执行目标函数
                if sync:
                    torch.cuda.synchronize(device_obj)
            
            # 正式计时运行
            times = []
            for i in range(warmup, repetitions + warmup):
                torch.cuda.synchronize(device_obj)
                start_events[i].record()
                
                result = func(*args, **kwargs)  # 执行目标函数
                
                end_events[i].record()
                if sync:
                    torch.cuda.synchronize(device_obj)
                
                # 计算本次执行时间 (毫秒)
                elapsed_time = start_events[i].elapsed_time(end_events[i])
                times.append(elapsed_time)
            
            # 计算统计指标
            times_np = np.array(times)
            avg_time = np.mean(times_np)
            min_time = np.min(times_np)
            max_time = np.max(times_np)
            
            # 计算TFLOPS和HBM带宽
            tflops = None
            hbm_bandwidth = None
            
            # 尝试自动检测矩阵乘法参数
            try:
                # 假设前两个参数是矩阵
                A, B = args[0], args[1]
                
                # 获取矩阵维度
                M, K = A.shape if len(A.shape) == 2 else (A.shape[0], np.prod(A.shape[1:]))
                K2, N = B.shape if len(B.shape) == 2 else (np.prod(B.shape[:-1]), B.shape[-1])
                
                # 验证矩阵乘法维度
                if K != K2:
                    print(f"Warning: Matrix dimensions mismatch: A[{M}x{K}] vs B[{K2}x{N}]")
                
                # 计算浮点运算次数 (2*M*N*K)
                flops = 2 * M * N * K
                
                # 计算内存访问量 (假设读写全部数据)
                # A: M*K, B: K*N, C: M*N
                # 数据类型大小
                dtype_size = A.element_size()
                memory_access = (M*K + K*N + M*N) * dtype_size
                
                # 计算TFLOPS (Tera FLoating-point OPerations per Second)
                tflops = (flops / 1e12) / (avg_time / 1000)  # 毫秒转秒
                
                # 计算HBM带宽 (GB/s)
                hbm_bandwidth = (memory_access / 1e9) / (avg_time / 1000)
                
            except (IndexError, AttributeError, TypeError) as e:
                print(f"Performance metrics calculation skipped: {e}")
            
            # 构建结果字典
            metrics = {
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'times_ms': times,
                'tflops': tflops,
                'hbm_bandwidth': hbm_bandwidth
            }
            
            # 打印结果
            print(f"Function '{func.__name__}' executed on cuda:{target_device}")
            print(f"  Warmup runs: {warmup}, Timed runs: {repetitions}")
            print(f"  Time: {avg_time:.4f} ms (min: {min_time:.4f}, max: {max_time:.4f})")
            
            if tflops is not None:
                print(f"  TFLOPS: {tflops:.2f}")
            if hbm_bandwidth is not None:
                print(f"  HBM Bandwidth: {hbm_bandwidth:.2f} GB/s")
            
            return result, metrics
        
        return wrapper
    return decorator

def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())

def get_build_sources():
    build_sources = []
    build_sources.append('bfgemm_cublas.cu')
    build_sources.append('pybind_bfgemm.cc')
    return build_sources

def get_project_dir():
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_build_cuda_cflags(build_pkg: bool = False):
    # -Xptxas -v:
    # registers, smem, cmem, stack, gmem usage
    # registers: 寄存器，访问速度最快。Ada Lovelace架构每个SM的寄存器文件大小
    # 为256KB，这相当于65536个32位寄存器，65536/256=256。一个SM可以同时执行多
    # 个block，对一个Kernel，同时存在于一个SM中的Block和Warp数量取决于SM中可用
    # 且所需的寄存器和共享内存数量。每个Thread需要的寄存器越多，那么SM中的Warp就
    # 越少。即减少Thread所需寄存器数量，即可增加SM中的Warp数。每个Block需要的共
    # 享内存越多，那么SM中可以被同时处理的Block就会变少。即减少每个Block所需的共
    # 享内存，即可同时处理更多Block。SM内的资源没办法处理一个完整Block，Kernel
    # 将无法启动。
    # cmem: 常量内存，被缓存，访问速度快。
    # stack frame: 由于寄存器的数量有限，当需要使用的变量数量超过可用寄存器数量时，
    # 编译器会将某些变量从寄存器“溢出”到栈上，这个过程称为spill。访问栈上的数据比
    # 访问寄存器慢得多。
    # spill stores: 指的是在执行过程中，数据因为寄存器不足而被存储到了栈上。
    # spill loads: 则是指将之前溢出到栈上的数据重新加载回寄存器。
    # diag 177: variable was declared but never referenced
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    if not build_pkg:
      extra_cuda_cflags.append("-diag-suppress 177")
      extra_cuda_cflags.append("-Xptxas -v")
    else:
      extra_cuda_cflags.append("--ptxas-options=-v")
      extra_cuda_cflags.append("--ptxas-options=-O3")
    # extra cuda flags for cute hgemm
    project_dir = get_project_dir()
    extra_cuda_cflags.append('-DNO_MMA_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_WMMA_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_CUTE_HGEMM_BIN')
    extra_cuda_cflags.append('-DNO_CUBLAS_HGEMM_BIN')
    # add cutlass headers and link cublas.
    
    extra_cuda_cflags.append('-lcublas')
    return extra_cuda_cflags


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


def build_from_sources(verbose: bool = False):
    torch_arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", None)         
    # Load the CUDA kernel as a python module
    pretty_print_line(f"Loading hgemm lib on device: {get_device_name()}, "
                      f"capability: {get_device_capability()}, "
                      f"Arch ENV: {torch_arch_list_env}")
    return load(name='hgemm_lib', sources=get_build_sources(),
                extra_cuda_cflags=get_build_cuda_cflags(), 
                extra_cflags=['-std=c++17'], 
                verbose=verbose)


def try_load_hgemm_library(force_build: bool = False, verbose: bool = False):
    if not force_build:
        # check if can import toy_hgemm
        try:
            import toy_hgemm as hgemm
            pretty_print_line(f"Import toy-hgemm library done, use it!")
        except Exception:
            pretty_print_line(f"Can't import toy-hgemm, force build "
                              f"from source or run <bash tools/install.sh>")
            pretty_print_line(f"Also may need export LD_LIBRARY_PATH="
                              f"PATH-TO/torch/lib:$LD_LIBRARY_PATH")
            hgemm = build_from_sources(verbose=verbose)
    else:
        pretty_print_line("Force hgemm lib build from sources")
        hgemm = build_from_sources(verbose=verbose)

    return hgemm


@torch.no_grad
def as_col_major(x: torch.Tensor):
    # convert a row major tensor -> col major with contiguous storage
    x_trans = x.t()
    x_col_major = x_trans.reshape(x.shape)
    return x_col_major.contiguous() # must be a contiguous tensor

