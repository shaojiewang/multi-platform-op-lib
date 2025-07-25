ARCH=sm_90a
rm -rf ./build
mkdir build
cd build
#/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH mma_inst.s -o kernel.co
nvcc -arch=$ARCH -std=c++17 -save-temps -Wno-deprecated-declarations -lcudart -lcuda ../mma_cuda.cu -o mma_cuda.exe

# run
chmod 777 ./mma_cuda.exe
./mma_cuda.exe wgmma_f32_64x128x16_bf16 64 128 16 1 16 1830
#./test v_mfma_f32_32x32x8_bf16 32 32 8 1 32 2100 
# disasm
cd -
cuobjdump -sass build/mma_cuda.exe > build/mma_cuda.sass

