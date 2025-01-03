.text
.global memcpy_matrix_loop_by_row_kernel_gfx942
.p2align 8
.type memcpy_matrix_loop_by_row_kernel_gfx942,@function
memcpy_matrix_loop_by_row_kernel_gfx942:
; This is just an example, not the optimal one
.set s_karg,            0   ; kernel argument
.set s_bx,              2   ; blockIdx

.set s_ptr_in,          4
.set s_ptr_out,         6
.set s_loops_per_block, 8
.set s_stride_block,    10
.set s_tmp,             12
.set s_k,               16
.set s_tilek,           20
.set s_log_tilek,       24
.set s_blk_stride,      28

.set v_buf,             0
.set v_offset,          16
.set v_tmp,             32

.set num_threads,       256
.set addr,              0xffffffff


    s_load_dwordx2 s[s_ptr_in:s_ptr_in+1],      s[s_karg:s_karg+1],     0
    s_load_dwordx2 s[s_ptr_out:s_ptr_out+1],    s[s_karg:s_karg+1],     8
    s_load_dword s[s_loops_per_block],          s[s_karg:s_karg+1],     16
    s_load_dword s[s_k],                        s[s_karg:s_karg+1],     20
    s_load_dword s[s_tilek],                    s[s_karg:s_karg+1],     24
    s_load_dword s[s_log_tilek],                s[s_karg:s_karg+1],     28
    s_load_dword s[s_blk_stride],               s[s_karg:s_karg+1],     32

    s_waitcnt       lgkmcnt(0)
    s_lshr_b32      s[s_tmp],   num_threads,    s[s_log_tilek]  ; blockDim / (tile_k / workload)
    s_lshl_b32      s[s_tmp],   s[s_tmp],       2
    s_mul_i32       s[s_tmp+1], s[s_tmp],       s[s_k]          ; block start address offset
    s_mul_i32       s[s_tmp+1], s[s_tmp+1],     s[s_bx]         ; blockIdx * offset

    s_lshr_b32      s[s_tilek], s[s_tilek],     2           ; tile_k / workload
    s_sub_u32       s[s_tmp],   s[s_tilek],     1
    v_and_b32       v[v_tmp],   v0,             s[s_tmp]    ; threadIdx % (tile_k/workload)
    v_lshrrev_b32   v[v_tmp+1], s[s_log_tilek], v0          ; threadIdx / tile_k
    v_lshlrev_b32   v[v_tmp+1], 2,              v[v_tmp+1]  ; threadIdx / tile_k * float

    v_lshl_add_u32      v[v_tmp],   v[v_tmp],   4,  s[s_tmp+1]  ; block offset + threadIdx % (tile_k/4) * float4
    v_mul_u32_u24       v[v_tmp+1], v[v_tmp+1], s[s_k]          ; threadIdx / tile_k * K * float
    v_add_u32           v[v_offset+0],   v[v_tmp],   v[v_tmp+1]

    s_lshl_b32 s[s_tmp],  s[s_blk_stride],  2            ; block_stride * float
    v_add_u32 v[v_offset+1],    s[s_tmp],   v[v_offset+0]
    v_add_u32 v[v_offset+2],    s[s_tmp],   v[v_offset+1]
    v_add_u32 v[v_offset+3],    s[s_tmp],   v[v_offset+2]

    s_lshl_b32 s[s_stride_block],   s[s_tmp],   2   ; unroll 4

    ;v_mov_b32 v[v_offset+3], addr
    ;global_load_dword v[v_buf+1],    v[v_offset+3],  s[s_ptr_in:s_ptr_in+1]
    ;s_waitcnt       vmcnt(0)

label_memcopy_start:
    global_load_dwordx4 v[v_buf+0 :v_buf+3],    v[v_offset+0],  s[s_ptr_in:s_ptr_in+1]
    global_load_dwordx4 v[v_buf+4 :v_buf+7],    v[v_offset+1],  s[s_ptr_in:s_ptr_in+1]
    global_load_dwordx4 v[v_buf+8 :v_buf+11],   v[v_offset+2],  s[s_ptr_in:s_ptr_in+1]
    global_load_dwordx4 v[v_buf+12:v_buf+15],   v[v_offset+3],  s[s_ptr_in:s_ptr_in+1]

    s_add_u32   s[s_ptr_in],   s[s_stride_block], s[s_ptr_in]
    s_addc_u32  s[s_ptr_in+1], s[s_ptr_in+1], 0

    s_waitcnt       vmcnt(0)    ; Counts the number of VMEM instructions issued but not yet completed

    global_store_dwordx4    v[v_offset+0],  v[v_buf+0 :v_buf+3],    s[s_ptr_out:s_ptr_out+1]
    global_store_dwordx4    v[v_offset+1],  v[v_buf+4 :v_buf+7],    s[s_ptr_out:s_ptr_out+1]
    global_store_dwordx4    v[v_offset+2],  v[v_buf+8 :v_buf+11],   s[s_ptr_out:s_ptr_out+1]
    global_store_dwordx4    v[v_offset+3],  v[v_buf+12:v_buf+15],   s[s_ptr_out:s_ptr_out+1]

    s_add_u32   s[s_ptr_out],   s[s_stride_block], s[s_ptr_out]
    s_addc_u32  s[s_ptr_out+1], s[s_ptr_out+1], 0

    s_sub_u32 s[s_loops_per_block], s[s_loops_per_block], 1
    s_cmp_eq_u32 s[s_loops_per_block], 0
    s_waitcnt       vmcnt(0)
    s_cbranch_scc0  label_memcopy_start
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel memcpy_matrix_loop_by_row_kernel_gfx942
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_accum_offset 64
    # .amdhsa_wavefront_size32 1
    # .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: memcpy_matrix_loop_by_row_kernel_gfx942
    .symbol: memcpy_matrix_loop_by_row_kernel_gfx942.kd
    .sgpr_count: 32
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: input,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: output,          .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: loops_per_block, .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32}
    - { .name: k,               .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32}
    - { .name: tilek,           .size: 4, .offset:  24, .value_kind: by_value, .value_type: i32}
    - { .name: log_tilek,       .size: 4, .offset:  28, .value_kind: by_value, .value_type: i32}
    - { .name: blk_stride,      .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
