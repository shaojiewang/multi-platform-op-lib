#!/bin/sh

KSRC=memcpy_x4_kernel_gfx90a.s
KOUT=memcpy_x4_kernel_gfx90a.hsaco
SRC=main.cpp
TARGET=out.exe

rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx942 $KSRC -o $KOUT

rm -rf $TARGET
/opt/rocm/bin/hipcc $SRC --offload-arch=gfx942 -o $TARGET
