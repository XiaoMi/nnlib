
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for things used internally.
 */

	.text
#if 0
static inline void biasadd_relu_requant_hvx(
	uint8_t *out,
	const int32_t *tmp_out,
	const int32_t *biasbuf,
	const uint32_t num_patches,
	const uint32_t depth,
	const uint32_t fixed_recip_level_size)
{
	int32_t sum;
	int32_t i,j;
	int32_t depth_in_vectors = depth / 32;
	int32_t outval;
	/* do 4 vectors */
	/* multiply */
	/* pack odd halves */
	/* saturate and pack */
	/* deal */
	/* deal */
	/* store a vector */
	for (j = 0; j < num_patches; j++) {
		for (i = 0; i < depth; i++) {
			sum = biasbuf[i] + tmp_out[j*depth+i];
			outval = sum * fixed_recip_level_size + (1<<15);
			outval >>= 16;
			if (outval < 0) outval = 0;
			if (outval > 255) outval = 255;
			*out++ = outval;
		}
	}
}
#endif

#define OUTPTR r0
#define INPTR r1
#define BIASBUF r2
#define NUM_PATCHES r3
#define DEPTH r4
#define RECIP_LEVEL_SIZE r5

#define BIASBUF_REWIND_ADDR r6
#define BIASPTR r7
#define ITERS r8
#define ROUNDAMT_R r9
#define L2FETCH_CTL r11:10
#define L2FETCH_CTL_LO r10
#define L2FETCH_CTL_HI r11

#define BUF0 v0
#define BUF1 v1
#define BUF2 v2
#define BUF3 v3

#define BBUF v9
#define VMPYAMT v8
#define HBUF1 v7
#define HBUF0 v6
#define BIASVALS v5

#define OUT0 v10
#define OUT1 v11
#define OUT2 v12
#define OUT3 v13
	.global biasadd_relu_requant_hvx
	.p2align 6
biasadd_relu_requant_hvx:
	{
		L2FETCH_CTL_HI = asl(DEPTH,#4)
		ITERS = mpyi(NUM_PATCHES,DEPTH)
	}
	{
		VMPYAMT = vsplat(RECIP_LEVEL_SIZE)
		ITERS = add(ITERS,#255)		// round up, +1 iter made soem mods for multithreading
		ROUNDAMT_R = ##0x08000
	}
	{
		ITERS = lsr(ITERS,#7)		// divide by 128
		BIASPTR = BIASBUF
		NUM_PATCHES = lsr(NUM_PATCHES,#2)	// 4x inputs...
	}
	{
		OUT0 = vsplat(ROUNDAMT_R)
		OUT1 = vsplat(ROUNDAMT_R)
		L2FETCH_CTL_LO = combine(L2FETCH_CTL_HI.L,NUM_PATCHES.L)
	}
	{
		OUT2 = vsplat(ROUNDAMT_R)
		OUT3 = vsplat(ROUNDAMT_R)
	}
	{
		l2fetch(INPTR,L2FETCH_CTL)
		BIASBUF_REWIND_ADDR = addasl(BIASBUF,DEPTH,#2)
	}
	{
		p3=sp1loop0(.Loop,ITERS)
	}
	.falign
.Loop:
	{
		BIASVALS = vmem(BIASPTR++#1)
		OUT0.w += vmpyie(BUF0.w,VMPYAMT.uh)
	}
	{
		BUF0 = vmem(INPTR++#1):nt
		p0 = cmp.eq(BIASBUF_REWIND_ADDR,BIASPTR)
		if (p0.new) BIASPTR = BIASBUF
		OUT1.w += vmpyie(BUF1.w,VMPYAMT.uh)
	}
	{
		BIASVALS = vmem(BIASPTR++#1)
		BUF0.w = vadd(BUF0.w,BIASVALS.w)
		OUT2.w += vmpyie(BUF2.w,VMPYAMT.uh)
	}
	{
		BUF1 = vmem(INPTR++#1):nt
		p0 = cmp.eq(BIASBUF_REWIND_ADDR,BIASPTR)
		if (p0.new) BIASPTR = BIASBUF
		OUT3.w += vmpyie(BUF3.w,VMPYAMT.uh)
	}
	{
		BIASVALS = vmem(BIASPTR++#1)
		OUT1 = vsplat(ROUNDAMT_R)
		BUF1.w = vadd(BUF1.w,BIASVALS.w)
		HBUF0.h = vpacko(OUT1.W,OUT0.W)
	}
	{
		BUF2 = vmem(INPTR++#1):nt
		p0 = cmp.eq(BIASBUF_REWIND_ADDR,BIASPTR)
		if (p0.new) BIASPTR = BIASBUF
		HBUF1.h = vpacko(OUT3.W,OUT2.W)
	}
	{
		BIASVALS = vmem(BIASPTR++#1)
		OUT2 = vsplat(ROUNDAMT_R)
		OUT3 = vsplat(ROUNDAMT_R)
		BUF2.w = vadd(BUF2.w,BIASVALS.w)
	}
	{
		BUF3.tmp = vmem(INPTR++#1):nt
		BUF3.w = vadd(BUF3.w,BIASVALS.w)
		p0 = cmp.eq(BIASBUF_REWIND_ADDR,BIASPTR)
		if (p0.new) BIASPTR = BIASBUF
	}
	{
		OUT0 = vsplat(ROUNDAMT_R)
		if (p3) vmem(OUTPTR++#1)=BBUF.new
		BBUF.ub = vpack(HBUF1.h,HBUF0.h):sat
	}:endloop0
	jumpr r31

#ifdef BIST

	.global main
main:
	r0 = syscfg
	r0 = setbit(r0,#7)
	syscfg = r0
	r0 = ssr
	r0 = setbit(R0,#31)
	r0 = setbit(r0,#29)
	ssr = r0
	r0 = ##out
	r1 = ##indata
	r2 = ##biasvals
	r3 = #4
	r4 = #32
	r5 = ##0x0000FFFF
	call biasadd_relu_requant_hvx
1:
	jump 1b

	.data
	.align 128
indata:
	.word 0
	.word 1
	.word 2
	.word 3
	.word 4
	.word 5
	.word 6
	.word 7
	.word 8
	.word 9
	.word 10
	.word 11
	.word 12
	.word 13
	.word 14
	.word 15
	.word 16
	.word 17
	.word 18
	.word 19
	.word 20
	.word 21
	.word 22
	.word 23
	.word 24
	.word 25
	.word 26
	.word 27
	.word 28
	.word 29
	.word 30
	.word 31
	.word 32
	.word 33
	.word 34
	.word 35
	.word 36
	.word 37
	.word 38
	.word 39
	.word 40
	.word 41
	.word 42
	.word 43
	.word 44
	.word 45
	.word 46
	.word 47
	.word 48
	.word 49
	.word 50
	.word 51
	.word 52
	.word 53
	.word 54
	.word 55
	.word 56
	.word 57
	.word 58
	.word 59
	.word 60
	.word 61
	.word 62
	.word 63
	.word 64
	.word 65
	.word 66
	.word 67
	.word 68
	.word 69
	.word 70
	.word 71
	.word 72
	.word 73
	.word 74
	.word 75
	.word 76
	.word 77
	.word 78
	.word 79
	.word 80
	.word 81
	.word 82
	.word 83
	.word 84
	.word 85
	.word 86
	.word 87
	.word 88
	.word 89
	.word 90
	.word 91
	.word 92
	.word 93
	.word 94
	.word 95
	.word 96
	.word 97
	.word 98
	.word 99
	.word 100
	.word 101
	.word 102
	.word 103
	.word 104
	.word 105
	.word 106
	.word 107
	.word 108
	.word 109
	.word 110
	.word 111
	.word 112
	.word 113
	.word 114
	.word 115
	.word 116
	.word 117
	.word 118
	.word 119
	.word 120
	.word 121
	.word 122
	.word 123
	.word 124
	.word 125
	.word 126
	.word 127

biasvals:
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0

out:
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0
	.word 0




#endif

