/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
      loop1(.Lpadzap_whole_rowloop,ROW_ITERS)       //
      VAL = vsplatb(VAL)                            //
    }
    {
      loop0(.Lpadzap_whole_d32loop,D32_ITERS)       //
      TMP = mpyi(D32_STRIDE,D32_ITERS)              //
      nop                                           //
    }
    {
      M0 = D32_STRIDE                               //
      VVAL = vsplat(VAL)                            //
      ROW_STRIDE = sub(ROW_STRIDE,TMP)              //
    }
.Lpadzap_whole_rowloop:
.Lpadzap_whole_d32loop:
    {
      vmem(PTR++M0):nt = VVAL                       //
    }:endloop0
    {
      loop0(.Lpadzap_whole_d32loop,D32_ITERS)       //
      PTR = add(PTR,ROW_STRIDE)                     //
    }:endloop1
    {
      r0 = #0                                       //
      jumpr r31                                     //
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
      VAL = vsplatb(VAL)                            // value to zap with
      W = memw(r29+#0)                              // width in D32
      VZERO = #0                                    //
      VMASK = vsplat(PTR)                           // guaranteed non NULL
    }
    {
      loop0(.Lpadzap_part_d32loop,D32_ITERS)        // inner loop setup
      PTR_OFF = and(PTR,#127)                       //
      PTR = and(PTR,#-128)                          //
      W = mpyi(W,#-32)                              // 128-W*32
    }
    {
      VVAL = vsplat(VAL)                            // value to zap with
      VMASK = valign(VZERO,VMASK,W)                 // insert W*32 bytes into low part
    }
    {
      M0 = D32_STRIDE                               //
      TMP = mpyi(D32_STRIDE,D32_ITERS)              //
      VMASK = vlalign(VMASK,VZERO,PTR_OFF)          // Move low part according to alignment
    }
    {
      loop1(.Lpadzap_part_rowloop,ROW_ITERS)        // outer loop setup
      QMASK = vcmp.gt(VMASK.uw,VZERO.uw)            // 1 if nonzero
      ROW_STRIDE = sub(ROW_STRIDE,TMP)              //
    }
.Lpadzap_part_rowloop:
.Lpadzap_part_d32loop:
    {
      if (QMASK) vmem(PTR++M0):nt = VVAL            //
    }:endloop0
    {
      loop0(.Lpadzap_part_d32loop,D32_ITERS)        //
      PTR = add(PTR,ROW_STRIDE)                     //
    }:endloop1
    {
      r0 = #0                                       //
      jumpr r31                                     //
    }
	.size padzap_part,.-padzap_part

//==================================================================================
#define start        r0
#define val          r1
#define d32_stride   r2
#define d32_iters    r3
#define row_stride   r4
#define row_iters    r5
#define width        r6
#define ptr_off2     width
#define ptr_off_w    r7
#define ptr_off      r8
#define minusone     r9
#define start_align  r10
#define start_align0 r11

#define mask0        q0
#define mask1        q1
#define sZero        v0
#define sMinusOne    v1
#define sVal         v2
#define sMask        v3

        .text
        .global padzap16_part
        .type padzap16_part,@function
        .p2align 6
padzap16_part:
    {
      val = combine(val.l,val.l)                    //
      width = memw(r29+#0)                          //
      ptr_off = and(start,#127)                     //
      d32_stride = add(d32_stride,d32_stride)       //
    }
    {
      minusone = #-1                                //
      start_align = and(start,#-128)                //
      ptr_off_w = addasl(ptr_off,width,#6)          //
      width = asl(width,#6)                         //
    }
    {
      sZero = #0                                    //
      sVal = vsplat(val)                            //
      sMinusOne = vsplat(minusone)                  //
      p0 = cmp.gtu(ptr_off_w,#128)                  //
    }
    {
      if p0 jump padzap16_part_2                    //
      width = neg(width)                            //
      m0 = d32_stride                               //
      row_stride = add(row_stride,row_stride)       //
    }
    {
      sMask = valign(sZero, sMinusOne, width)       //
      ptr_off2 = ptr_off                            //
    }
    {
      sMask = vlalign(sMask, sZero, ptr_off2)       //
      loop1(padzap16_part_A_lp,row_iters)           //
    }
    {
      mask0 = vcmp.gt(sMask.ub, sZero.ub)           //
      loop0(padzap16_part_A_lp,d32_iters)           //
      start_align0 = add(start_align,row_stride)    //
    }
padzap16_part_A_lp:
    {
      if (mask0) vmem(start_align++m0):nt = sVal    //
    }:endloop0
    {
      loop0(padzap16_part_A_lp,d32_iters)           //
      start_align = start_align0                    //
      start_align0 = add(start_align0,row_stride)   //
    }:endloop1
    {
      jumpr r31                                     //
    }

        .p2align 6
padzap16_part_2:
    {
      ptr_off_w = neg(ptr_off_w)                    //
      mask0 = vsetq(ptr_off)                        //
      p0 = cmp.eq(row_stride,d32_stride)            //
    }
    {
      sMask = valign(sZero, sMinusOne, ptr_off_w)   //
      loop1(padzap16_part_B_lp,row_iters)           //
      if (p0) jump padzap16_part_3                  //
    }
    {
      mask0 = not(mask0)                            //
      mask1 = vcmp.gt(sMask.ub,sZero.ub)            //
      loop0(padzap16_part_B_lp,d32_iters)           //
      start_align0 = add(start_align,row_stride)    //
    }
padzap16_part_B_lp:
    {
      if (mask1) vmem(start_align+#1):nt = sVal     //
    }
    {
      if (mask0) vmem(start_align++m0):nt = sVal    //
    }:endloop0
    {
      loop0(padzap16_part_B_lp,d32_iters)           //
      start_align = start_align0                    //
      start_align0 = add(start_align0,row_stride)   //
    }:endloop1
    {
      jumpr r31                                     //
    }

        .p2align 6
padzap16_part_3:
    {
      mask0 = not(mask0)                            //
      mask1 = vcmp.gt(sMask.ub,sZero.ub)            //
      loop0(padzap16_part_C_lp,row_iters)           //
      start_align0 = add(start_align,row_stride)    //
    }
padzap16_part_C_lp:
    {
      if (mask1) vmem(start_align+#1):nt = sVal     //
    }
    {
      if (mask0) vmem(start_align++m0):nt = sVal    //
    }:endloop0
    {
      jumpr r31                                     //
    }

        .size padzap16_part,.-padzap16_part