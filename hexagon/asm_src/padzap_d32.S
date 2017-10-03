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
 * Pad Zapping for D32 
 */

/*
 * Pad Zap Top/Bottom is just vmemset_asm
 */

/*
 * padzap_whole(uint8_t *start, uint8_t val, int d32_stride, int d32_iters, int row_stride, int row_iters)
 */

#define PTR r0
#define VAL r1
#define D32_STRIDE r2
#define D32_ITERS r3
#define ROW_STRIDE r4
#define ROW_ITERS r5
#define TMP r28

#define VVAL v0
	.text
	.global padzap_whole
	.type padzap_whole,@function
	.p2align 6
padzap_whole:
	{
		loop1(.Lpadzap_whole_rowloop,ROW_ITERS)
		VAL = vsplatb(VAL)
	}
	{
		loop0(.Lpadzap_whole_d32loop,D32_ITERS)
		TMP = mpyi(D32_STRIDE,D32_ITERS)
		nop
	}
	{
		M0 = D32_STRIDE
		VVAL = vsplat(VAL)
		ROW_STRIDE = sub(ROW_STRIDE,TMP)
	}
.Lpadzap_whole_rowloop:
.Lpadzap_whole_d32loop:
	{
		vmem(PTR++M0):nt = VVAL
	}:endloop0
	{
		loop0(.Lpadzap_whole_d32loop,D32_ITERS)
		PTR = add(PTR,ROW_STRIDE)
	}:endloop1
	{
		r0 = #0
		jumpr r31
	}
	.size padzap_whole,.-padzap_whole

/*
 * padzap_part(uint8_t *start, uint8_t val, int d32_stride, int d32_iters, int row_stride, int row_iters, int w)
 * Zap 32*W elements starting @ start, which might not be aligned.
 * It is not allowed to span to a new vector.
 */


#define PTR r0
#define VAL r1
#define D32_STRIDE r2
#define D32_ITERS r3
#define ROW_STRIDE r4
#define ROW_ITERS r5
#define W r6
#define PTR_OFF r7
#define TMP r28

#define VVAL v0
#define VMASK v1
#define VZERO v2
#define QMASK q0

	.text
	.global padzap_part
	.type padzap_part,@function
	.p2align 6
padzap_part:
	{
		loop1(.Lpadzap_part_rowloop,ROW_ITERS)		// outer loop setup
		VAL = vsplatb(VAL)				// value to zap with
		W = memw(r29+#0)				// width in D32
	}
	{
		loop0(.Lpadzap_part_d32loop,D32_ITERS)		// inner loop setup
		PTR_OFF = and(PTR,#127)
		PTR = and(PTR,#-128)
		W = asl(W,#5)					// bytes of width
	}
	{
		VVAL = vsplat(VAL)				// value to zap with
		VMASK = vsplat(PTR)				// guaranteed non NULL
		VZERO = #0
		W = sub(#128,W)					// 128-width_in_bytes
	}
	{
		VMASK = valign(VZERO,VMASK,W)			// insert W*32 bytes into low part
		M0 = D32_STRIDE
		TMP = mpyi(D32_STRIDE,D32_ITERS)
	}
	{
		VMASK = vlalign(VMASK,VZERO,PTR_OFF)		// Move low part according to alignment
		ROW_STRIDE = sub(ROW_STRIDE,TMP)
	}
	{
		QMASK = vcmp.gt(VMASK.uw,VZERO.uw)		// 1 if nonzero
	}
.Lpadzap_part_rowloop:
.Lpadzap_part_d32loop:
	{
		if (QMASK) vmem(PTR++M0):nt = VVAL
	}:endloop0
	{
		loop0(.Lpadzap_part_d32loop,D32_ITERS)
		PTR = add(PTR,ROW_STRIDE)
	}:endloop1
	{
		r0 = #0
		jumpr r31
	}
	.size padzap_part,.-padzap_part



