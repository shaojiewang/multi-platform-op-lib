ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH mma_inst.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  test.cu -o test

# run
./test v_mfma_f32_16x16x16_bf16 16 16 16 1 16 1410

