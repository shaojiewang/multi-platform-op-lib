ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH  lds_op.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  run_asm.cpp -o bench.exe


# run
 ./bench.exe v_mfma_f32_16x16x1f32 16 16 1  1 64 1
