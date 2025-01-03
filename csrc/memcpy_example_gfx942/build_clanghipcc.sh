#!/bin/sh

SRC=test.cpp
TARGET=out

for KSRC in *.s; do
    if [ -f "$KSRC" ]; then
        KOUT="${KSRC%.s}.hsaco"
        rm -rf $KOUT

        /opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=gfx942:sramecc+:xnack- "$KSRC" -o "$KOUT"
    fi
done
rm -rf $TARGET
/opt/rocm/bin/hipcc $SRC --offload-arch=gfx942:sramecc+:xnack- -o $TARGET

