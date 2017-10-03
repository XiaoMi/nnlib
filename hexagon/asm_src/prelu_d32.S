/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#define IN_OUT r1:0
#define OUT r0
#define IN r1
#define IN_NEXT_ROW r2
#define IN_NEXT_D32 r3
#define OUT_NEXT_ROW IN_NEXT_ROW	// assume they are the same for now
#define OUT_NEXT_D32 IN_NEXT_D32	// assume they are the same for now
#define QZERO_ALPHABUF r5:4
#define QZERO r5
#define ALPHABUF r4
#define ALPHA_SHIFT r6
#define RSHIFT7 r7
#define D_D32_ITERS r9:8
#define D_ITERS r9
#define D32_ITERS r8
#define NEXT_OUTER_IN r11
#define NEXT_OUTER_OUT r10
#define NEXT_IN r13
#define NEXT_OUT r12
#define H_ITERS r14
#define SHRINK r15
#define SHRINK_H_ITERS r15:14
#define OUT_QZERO r28


#define VQZERO v0
#define INVALS v1
#define NEGOFFS v2
//#define ZOFFS_OUT v3
#define PRODUCTS v5:4
#define PRODUCTSHI v5
#define PRODUCTSLO v4
#define ALPHAS v6
#define DONEVALS v7
#define POSOFFS_OUT v8
#define NEGOFFS_OUT v9

#define POSPRODS v11:10
#define POSPRODSHI v11
#define POSPRODSLO v10
#define POSOFFS v12
#define VZERO v13
#define VOUT_QZERO v14
#define VSHRINK v15
#define VALPHA_SHRINK v15





	.text
	.global prelu_hvx_d32
	.type prelu_hvx_d32,@function
	.p2align 6
prelu_hvx_d32:
	{
		D_D32_ITERS = memd(r29+#0)
		SHRINK_H_ITERS = memd(r29+#8)
		QZERO = vsplatb(QZERO)
		RSHIFT7 = #7
	}
	{
		OUT_QZERO = memw(r29+#16)
		ALPHA_SHIFT = memw(r29+#20)
		VZERO = vxor(VZERO,VZERO)
		SHRINK = vsplatb(SHRINK)
	}
	{
		//D32_ITERS = add(D32_ITERS,#1)
		NEXT_OUTER_IN = add(IN,IN_NEXT_D32)
		NEXT_OUTER_OUT = add(OUT,OUT_NEXT_D32)
		OUT_QZERO = vsplatb(OUT_QZERO)
	}
	{
		ALPHAS = vmem(ALPHABUF++#1)
		loop1(.Lh_iter,H_ITERS)
		VQZERO = vsplat(QZERO)
		ALPHA_SHIFT = sub(#7,ALPHA_SHIFT)
	}
	{
		VSHRINK = vsplat(SHRINK)
		NEXT_IN = add(IN,IN_NEXT_ROW)
		NEXT_OUT = add(OUT,OUT_NEXT_ROW)
	}
	{
		//p3=sp1loop0(.Looptop,D32_ITERS)
		loop0(.Looptop,D32_ITERS)
		p3 = cmp.eq(r0,r0)
		VOUT_QZERO = vsplat(OUT_QZERO)
	}
	{
		PRODUCTS.h = vmpy(VSHRINK.ub,ALPHAS.b)
	}
	{
		ALPHAS.b = vasr(PRODUCTSHI.h,PRODUCTSLO.h,ALPHA_SHIFT):rnd:sat
	}
	.falign
.Ld32_iter:
.Lh_iter:
.Looptop:
	{
		INVALS.tmp = vmem(IN++#1)			// get input data // 1st
		NEGOFFS.ub = vsub(VQZERO.ub,INVALS.ub):sat	// get negative offsets, pos-->0 // 1st
		POSOFFS.ub = vsub(INVALS.ub,VQZERO.ub):sat	// get pos offsets, neg-->0 // 1st
	}
	{
		POSPRODS.uh = vmpy(POSOFFS.ub,SHRINK.ub)	// scale to output // 1st
	}
	{
		PRODUCTS.h = vmpy(NEGOFFS.ub,ALPHAS.b)		// multiply by alpha & scale // 1st
	}
		POSOFFS_OUT.ub = vasr(POSPRODSHI.h,POSPRODSLO.h,RSHIFT7):rnd:sat
		NEGOFFS_OUT.b = vasr(PRODUCTSHI.h,PRODUCTSLO.h,RSHIFT7):rnd:sat
		POSOFFS_OUT.ub = vadd(VOUT_QZERO.ub,POSOFFS_OUT.ub):sat
	{
		DONEVALS.b = vsub(POSOFFS_OUT.b,NEGOFFS_OUT.b)	// hopefully our math is right 
		//if (p3) vmem(OUT++#1) = DONEVALS.new
		vmem(OUT++#1) = DONEVALS.new
	}


#if 0
	{
		INVALS.TMP = vmem(IN++#1)				// 1st iter
		ZOFFS_IN.b = vsub(VQZERO.b,INVALS.b)			// 1st iter
		POSOFFS2.h = vsub(INVALS.ub,VQZERO.ub)			// 1st iter, DOUBLE RESOURCE
	}
	{
		PRODUCTS.h = vmpy(ZOFFS_IN.ub,ALPHAS.b)			// 1st iter, DOUBLE RESOURCE
		POSOFFS.ub = vasr(POSOFFS2HI.h,POSOFFS2LO.h,POS_SHIFT_R):rnd:sat	// 1st iter, negative values --> 0
	}
		ZOFFS_OUT.b = vasr(PRODUCTSHI.h,PRODUCTSLO.h,RSHIFT):rnd:sat	//2nd
	{
		OUTVALS.b = vsub(VQZERO.b,ZOFFS_OUT.b)			// 2nd
		QSMALLS = vcmp.eq(VZERO.b,POSOFFS.b)			// 2nd iter, > 0 offset?
		POSOFFS.b = vadd(VQZERO.b,POSOFFS.b)			// 2nd iter
	}
		DONEVALS = vmux(QSMALLS,OUTVALS,POSOFFS)		// 2nd
		if (p3) vmem(OUT++#1) = DONEVALS.new			// 2nd
#endif
	{ nop }:endloop0
	{
		IN_OUT = combine(NEXT_IN,NEXT_OUT)
		NEXT_IN = add(NEXT_IN,IN_NEXT_ROW)
		NEXT_OUT = add(NEXT_OUT,OUT_NEXT_ROW)
		loop0(.Looptop,D32_ITERS)
		//p3=sp1loop0(.Looptop,D32_ITERS)
	}:endloop1
	{
		IN_OUT = combine(NEXT_OUTER_IN,NEXT_OUTER_OUT)
		NEXT_OUTER_IN = add(NEXT_OUTER_IN,IN_NEXT_D32)
		NEXT_OUTER_OUT = add(NEXT_OUTER_OUT,OUT_NEXT_D32)
		ALPHAS = vmem(ALPHABUF++#1)
	}
	{
		NEXT_IN = add(IN,IN_NEXT_ROW)
		NEXT_OUT = add(OUT,OUT_NEXT_ROW)
		PRODUCTS.h = vmpy(VSHRINK.ub,ALPHAS.b)
		D_ITERS = add(D_ITERS,#-1)
	}
	{
		loop1(.Looptop,H_ITERS)
		p0 = cmp.eq(D_ITERS,#0)
		if (!p0.new) jump:t .Lh_iter
		ALPHAS.b = vasr(PRODUCTSHI.h,PRODUCTSLO.h,ALPHA_SHIFT):rnd:sat
	}
	jumpr r31
	.size prelu_hvx_d32,.-prelu_hvx_d32


#if 0
	/* prelu_hvx_large_d32(out,in,in_next_row,in_next_d32,w_iters,d32_iters,height,alpha_frac_buf,qzero,shift) */
	.text
	.global prelu_hvx_large_d32
	.type prelu_hvx_large_d32,@function
	.p2align 6
prelu_hvx_large_d32:
	{
		D_D32_ITERS = memd(r29+#0)
		POS_SHIFT_H_ITERS = memd(r29+#8)
		QZERO = vsplatb(QZERO)
		RSHIFT = #7
	}
	{
		D32_ITERS = add(D32_ITERS,#1)
		NEXT_OUTER_IN = add(IN,IN_NEXT_D32)
		NEXT_OUTER_OUT = add(OUT,OUT_NEXT_D32)
		VZERO = vxor(VZERO,VZERO)
	}
	{
		ALPHAS = vmem(ALPHABUF++#1)
		loop1(.Ll_h_iter,H_ITERS)
		VQZERO = vsplat(QZERO)
	}
	{
		p3=sp1loop0(.Looptop_l,D32_ITERS)
		NEXT_IN = add(IN,IN_NEXT_ROW)
		NEXT_OUT = add(OUT,OUT_NEXT_ROW)
		POS_SHIFT_R = POS_SHIFT
	}
	.falign
.Ll_d32_iter:
.Ll_h_iter:
.Looptop_l:
	{
		INVALS.TMP = vmem(IN++#1)				// 1st iter
		ZOFFS_IN.b = vsub(VQZERO.b,INVALS.b)			// 1st iter, DOUBLE RESOURCE
		POSOFFS2.h = vsub(INVALS.ub,VQZERO.ub)			// 1st iter
		ZOFFS_OUT.b = vasr(PRODUCTSHI.h,PRODUCTSLO.h,RSHIFT):rnd:sat	//2nd
	}
	{
		OUTVALS.b = vsub(VQZERO.b,ZOFFS_OUT.b)			// 2nd
		QSMALLS = vcmp.eq(VZERO.b,POSOFFS.b)			// 2nd iter, > 0 offset?
		POSOFFS.b = vadd(VQZERO.b,POSOFFS.b)			// 2nd iter
	}
	{
		PRODUCTS.h = vmpy(ZOFFS_IN.ub,ALPHAS.b)			// 1st iter, DOUBLE RESOURCE
		DONEVALS = vmux(QSMALLS,OUTVALS,POSOFFS)		// 2nd
		if (p3) vmem(OUT++#1) = DONEVALS.new			// 2nd
		POSOFFS.ub = vasr(POSOFFS2HI.h,POSOFFS2LO.h,POS_SHIFT_R):rnd:sat	// 1st iter, negative values --> 0
	}:endloop0
	{
		IN_OUT = combine(NEXT_IN,NEXT_OUT)
		NEXT_IN = add(NEXT_IN,IN_NEXT_ROW)
		NEXT_OUT = add(NEXT_OUT,OUT_NEXT_ROW)
		//loop0(.Looptop,D32_ITERS)
		p3=sp1loop0(.Looptop_l,D32_ITERS)
	}:endloop1
	{
		IN_OUT = combine(NEXT_OUTER_IN,NEXT_OUTER_OUT)
		NEXT_OUTER_IN = add(NEXT_OUTER_IN,IN_NEXT_D32)
		NEXT_OUTER_OUT = add(NEXT_OUTER_OUT,OUT_NEXT_D32)
		ALPHAS = vmem(ALPHABUF++#1)
	}
	{
		NEXT_IN = add(NEXT_IN,IN_NEXT_ROW)
		NEXT_OUT = add(NEXT_OUT,OUT_NEXT_ROW)
	}
	{
		loop1(.Looptop_l,H_ITERS)
		D_ITERS = add(D_ITERS,#-1)
		p0 = cmp.eq(D_ITERS,#1)
		if (!p0.new) jump:t .Lh_iter
	}
	jumpr r31
	.size prelu_hvx_d32,.-prelu_hvx_d32


#endif
