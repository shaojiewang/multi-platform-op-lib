ARCH=gfx942
rm -rf ./build
mkdir build
cd build
#/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH mma_inst.s -o kernel.co
/opt/rocm/bin/hipcc -x hip -O3 -Wno-deprecated-declarations -save-temps --offload-arch=$ARCH  ../mma_gpu.cc -o mma_gpu.exe

# run
./mma_gpu.exe v_mfma_f32_16x16x16_bf16 16 16 16 1 16 2100 
#./test v_mfma_f32_32x32x8_bf16 32 32 8 1 32 2100 
cd -
