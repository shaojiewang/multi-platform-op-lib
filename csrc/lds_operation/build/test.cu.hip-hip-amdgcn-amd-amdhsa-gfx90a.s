	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx90a"
	.section	.text._Z11test_kernelILi256ELi4096EEvPfS0_,"axG",@progbits,_Z11test_kernelILi256ELi4096EEvPfS0_,comdat
	.protected	_Z11test_kernelILi256ELi4096EEvPfS0_ ; -- Begin function _Z11test_kernelILi256ELi4096EEvPfS0_
	.globl	_Z11test_kernelILi256ELi4096EEvPfS0_
	.p2align	8
	.type	_Z11test_kernelILi256ELi4096EEvPfS0_,@function
_Z11test_kernelILi256ELi4096EEvPfS0_:   ; @_Z11test_kernelILi256ELi4096EEvPfS0_
; %bb.0:                                ; %entry
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	v_lshl_add_u32 v6, s6, 8, v0
	v_mov_b32_e32 v7, 0
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s0, s8
	s_mov_b32 s1, s9
	;;#ASMSTART
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen offset:0
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v2, offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v3, offset:0x100
	;;#ASMEND
	v_lshlrev_b64 v[6:7], 4, v[6:7]
	;;#ASMSTART
	ds_write_b32 v0, v4, offset:0x200
	;;#ASMEND
	v_mov_b32_e32 v8, s11
	v_add_co_u32_e32 v6, vcc, s10, v6
	v_lshlrev_b32_e32 v1, 2, v0
	;;#ASMSTART
	ds_write_b32 v0, v5, offset:0x300
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v1, offset:0
	;;#ASMEND
	v_addc_co_u32_e32 v7, vcc, v8, v7, vcc
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	global_store_dwordx4 v[6:7], v[2:5], off
	s_barrier
	;;#ASMSTART
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen offset:0x400
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v2, offset:0x400
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v3, offset:0x500
	;;#ASMEND
	s_movk_i32 s4, 0x2000
	;;#ASMSTART
	ds_write_b32 v0, v4, offset:0x600
	;;#ASMEND
	v_add_co_u32_e32 v8, vcc, s4, v6
	;;#ASMSTART
	ds_write_b32 v0, v5, offset:0x700
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v1, offset:0x400
	;;#ASMEND
	v_addc_co_u32_e32 v9, vcc, 0, v7, vcc
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	global_store_dwordx4 v[8:9], v[2:5], off offset:-4096
	s_barrier
	;;#ASMSTART
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen offset:0x800
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v2, offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v3, offset:0x900
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v4, offset:0xa00
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v5, offset:0xb00
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v1, offset:0x800
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	global_store_dwordx4 v[8:9], v[2:5], off
	s_barrier
	;;#ASMSTART
	buffer_load_dwordx4 v[2:5], v0, s[0:3], 0 offen offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v2, offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v3, offset:0xd00
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v4, offset:0xe00
	;;#ASMEND
	;;#ASMSTART
	ds_write_b32 v0, v5, offset:0xf00
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[0:3], v1, offset:0xc00
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_add_co_u32_e32 v4, vcc, 0x3000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	global_store_dwordx4 v[4:5], v[0:3], off
	s_barrier
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11test_kernelILi256ELi4096EEvPfS0_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_kernarg_preload_length  0
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 12
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z11test_kernelILi256ELi4096EEvPfS0_,"axG",@progbits,_Z11test_kernelILi256ELi4096EEvPfS0_,comdat
.Lfunc_end0:
	.size	_Z11test_kernelILi256ELi4096EEvPfS0_, .Lfunc_end0-_Z11test_kernelILi256ELi4096EEvPfS0_
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 416
; NumSgprs: 16
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 10
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_2dc02b19ac8d7768,@object ; @__hip_cuid_2dc02b19ac8d7768
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2dc02b19ac8d7768
__hip_cuid_2dc02b19ac8d7768:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2dc02b19ac8d7768, 1

	.ident	"AMD clang version 18.0.0git (ssh://gerritgit/lightning/ec/llvm-project amd-mainline-open 24165 b8a2fdd0a02f77c0a37740478912d2de24c20604)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2dc02b19ac8d7768
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           input.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11test_kernelILi256ELi4096EEvPfS0_
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z11test_kernelILi256ELi4096EEvPfS0_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
