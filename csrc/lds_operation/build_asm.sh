ARCH=gfx942
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=$ARCH  lds_op.s -o kernel.co
/opt/rocm/bin/hipcc --offload-arch=$ARCH  run_asm.cpp -o bench.exe


# run
./bench.exe 1000 1410
