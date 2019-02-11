/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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

 	.text
	.file	"vmemset_2d_h.S"

//
// HVX rectangle fill operation
//
// void vmemset_32_2d_asm(
//      void * dst,         // location
//      int val,            // 32_bit value to fill
//      int width,          // width of rectangle (bytes), >0
//      int height,         // height of rectangle; rows > 0
//      int stride );       // stride of buffer; must be multiple of vector
//
// This operation does any rectangle fill using vector operations; it uses masked
// writes as needed to avoid over-writing portions of the output buffer outside the
// specified area.
// The row pitches must be vector aligned (so whatever strategy is used on one row,
// is used on all others). When the width is small - up to 6 vectors per row -
// there is no 'horizontal' loop, just the height loop.
//
//
// The row-pitch condition is not checked. The wid >0, ht >0 condition is checked, and
// the routine does nothing if this is not the case.
//
// This will be a little smaller/faster for __HEXAGON_ARCH__ >= 62
// (uses vsetq2 for masks)
//
// For a general byte fill, the 'val' should be the same in all 4 byte lanes. (An inline wrapper vmemset_2d_asm can do this)
// If you want a 16-bit fill, put the value in both halves of 'value', and use an even width;
// the l.s. byte of the value will always be stored at the start of each row of the array, even if the pointer
// is odd.
//// Likewise, for 32-bit fill, supply the value as 'val'; the lsb of the value will be aligned to the start
// of each row (and 'width' should be a multiple of 4).
//
// There is also vmemset_2d_32_general_asm
// .. which has exactly the same form but no restriction on dst pitch.
// It will make multiple calls to vmemset_32_2d_asm as needed; for instance
// if the dst pitch is a multiple of 1/4 vector (but not 1/2) it will
// make 4 calls, each of which fills every 4th row.
//

#ifndef __HVXDBL__
#define VECN 64
#define VECLOGN 6
#else
#define VECN 128
#define VECLOGN 7
#endif
// r0 ->  dstp
// r1 -> fillval
// r2 -> width
// r3 -> height
// r4 -> dst_stride
#define dstp         r0
#define fillval      r1
#define width        r2
#define height       r3
#define dst_stride   r4
#define dstp2        r5
#define startoff     r7
#define endpos       r8
#define middlecount  r9
#define tmp          r10


#define qnLeft        q0
#define qRight       q1

#define vFill        v0


	.globl	vmemset_32_2d_asm
	.balign 16
	.type	vmemset_32_2d_asm,@function


vmemset_32_2d_asm:
   {
     startoff = and(dstp,#VECN-1);          // destination offset
     qnLeft = vsetq(dstp);                  // the 'left' mask
     vFill = vsplat(fillval);
   }
   // set qnLeft up to be  1 1 1 .. 0 0 0
   // to be the 'start' mask for (inverted) the first output vector in each row;
   //  and qRight to  1 1 1 .. 0 0 0 to be the 'end' mask for the last vector.
   // If there is only one output vector per row we'll and them later.
   {
     p0 = cmp.gt(width,#0)                  // protect against ht <=0, wid <= 0
     p0 = cmp.gt(height,#0)
     //dstp = sub(dstp,startoff);           // align the dest address (not needed)
     endpos = add(startoff,width);          // start off + wid: if this > VECN, needs >1 vector write per row
   } {
     if(!p0) jumpr:nt r31;                  // done if ht <= 0 or wid <= 0
#if __HEXAGON_ARCH__ < 62
     tmp = and(endpos,#VECN-1)              // test if end falls on a boundary
     qRight  = or(qnLeft,!qnLeft);          // the 'end' mask (last partial write - all 1's if falls on a boundary).
   } {
     p1 = cmp.eq(tmp,#0);	//
     if( p1.new) jump:nt .L_mskskip;
     m0 = dst_stride;                      // stride
   } {
     qRight = vsetq(endpos);               // this the 'end' mask in the general case.
   }
.L_mskskip:
#else
     qRight = vsetq2(endpos);
     m0 = dst_stride;
   }
#endif
   {
     vFill = vlalign( vFill,vFill,startoff);  // adjust 16- or 32-bit value for pointer addr.
     // test if > 1 output vector per row is needed.
     endpos = add(endpos,#-(VECN+1));       // for finding middlecount
     P0=!cmp.gt(r8,#VECN);                  // true if single vector per row
     if( P0.new) jump:nt .L_row_onevec;     // r8 >= 0 if not taken.
   } {
     p0 = !cmp.gt(endpos,#VECN-1)           // true if 2 vectors per row
     loop0(.L_loop_twovec,height);
     dstp2 = add(dstp,#VECN)                // needed in the loop
     if( p0.new ) jump:nt .L_row_twovec;
   } {
     p0 = tstbit(endpos,#VECLOGN)           // p0 is true if middlecount is odd #
     p2 = cmp.gt( endpos, #VECN*5-1)        // true if middlecount  >= 5
     p3 = cmp.gt( endpos, #VECN*3-1)        // true if middlecount  >= 3
   } {
     if(p2) jump:nt .L_handle_general_outer;	// go handle the middlecount >=5 situation
     if(p3) jump:nt .L_handle_general_outer_34; // middlecount = 3 or 4
   } {
     loop0(.L_loop_general_outer_12,height);
   }
// this loop is used when them middlecount is 1 or 2
//
.balign 16
.L_loop_general_outer_12:
   {
     if( !qnLeft ) vmem(dstp++m0) = vFill;
     dstp2 = add(dstp,#VECN);
   } {
     if(!p0)vmem(dstp2++#1) = vFill;            // do this if middlecount is 2.
   } {
     vmem(dstp2++#1) = vFill;
   } {
     if( qRight ) vmem(dstp2+#0) = vFill;
   }:endloop0;
   { jumpr r31; }                         // !! all done

// this loop is used when the middlecount is 3 or 4
//
.balign 16
.L_handle_general_outer_34:
   {
     loop0(.L_loop_general_outer_34,height);
   }
.balign 16
 .L_loop_general_outer_34:
   {
     if( !qnLeft ) vmem(dstp++m0) = vFill;
     dstp2 = add(dstp,#VECN);
   } {
     if(!p0)vmem(dstp2++#1) = vFill;            // only when 4.
   } {
     vmem(dstp2++#1) = vFill;
   } {
     vmem(dstp2++#1) = vFill;
   } {
     vmem(dstp2++#1) = vFill;
   }
   {
     if( qRight ) vmem(dstp2+#0) = vFill;
   }:endloop0;
    { jumpr r31; }                              // !! all done

// This loop is used when the 'inner' vectors are at least 5.
// A lot of strangeness here to avoid having a short inner loop with a low loop count
// inside an outer loop with a large loop count...
//
// inner part consists of:
//   - unconditional store
//   - conditional store #1
//   - conditional store pair #2
//   - loop of 4 stores
// So the plan is like this (mc = actual middle count)
//
//   mc    cond#1    cond#2     innerloops
//   5       0       0                1
//   6       1       0                1
//   7       0       1                1
//   8       1       1                1
//   9       0       0                2
//   10      1       0                2
// etc.
// So the "conditional store#1 is done when p0 is false, we already have that.
// To get the rest:
//  (1) subtract VECN from endpos; bit VECLOGN+1 will give the condition for #2 in p1
//  (2) >> that result by VECLOGN+2 to get the innerloop count.
//
//
.L_handle_general_outer:
   {
      tmp = add(endpos, #-VECN );
      loop1(.L_loop_general_outer,height );
   } {
      middlecount = lsr(tmp,#VECLOGN+2)         // # actually (middlecount-1) /4
      p1 = tstbit(tmp,#VECLOGN+1)               // p1 is true if actual middlecount is 4*k or 4*k+3
   } {
     loop0(.L_loop_general_inner,middlecount );
   }
.balign 16
 .L_loop_general_outer:
   {
     if( !qnLeft ) vmem(dstp++m0) = vFill;
     dstp2 = add(dstp,#VECN);
   } {
     vmem(dstp2++#1) = vFill;
   } {
     if(!p0)vmem(dstp2++#1) = vFill;
   } {
     if(p1)vmem(dstp2++#1) = vFill;
   } {
     if(p1)vmem(dstp2++#1) = vFill;
   }
.balign 16
 .L_loop_general_inner:
   {
     vmem(dstp2++#1) = vFill;
   } {
     vmem(dstp2++#1) = vFill;
   } {
     vmem(dstp2++#1) = vFill;
   } {
     vmem(dstp2++#1) = vFill;
   }:endloop0;
   {
     loop0(.L_loop_general_inner,middlecount );
     if( qRight ) vmem(dstp2+#0) = vFill;
   }:endloop1;
   { jumpr r31; }                         // !! all done


////////////////////////////////////////////////////////////////////////////////////////////
// two vectors per row
//
.balign 16
.L_row_twovec:  // only two output vector per row...

.L_loop_twovec:
   {
     if( !qnLeft ) vmem(dstp++m0) = vFill;
   } {
     if( qRight ) vmem(dstp2++m0) = vFill;
   }:endloop0;
   { jumpr r31; }                         // !! all done

////////////////////////////////////////////////////////////////////////////////////////////
// only one vector per row
//
.balign 16
.L_row_onevec: // only one output vector per row...
   {
     loop0(.L_loop_onevec,height);          // set up 1-vector loop
     qnLeft = or(qnLeft,!qRight);        // make combined mask
   }
.balign 16
.L_loop_onevec:
   {
     if( !qnLeft ) vmem(dstp++m0) = vFill;
   }:endloop0;
   { jumpr r31; }                         // !! all done


.LtmpX:
   .size	vmemset_32_2d_asm, .LtmpX-vmemset_32_2d_asm

#undef dstp
#undef fillval
#undef width
#undef height
#undef dst_stride
#undef dstp2
#undef startoff
#undef endpos
#undef middlecount
#undef tmp
#undef qnLeft
#undef qRight
#undef vFill

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
// General vmemset_2D, which allows any rowpitch,
// by calling the other copy multiple times as needed.
// The better the alignment of the supplied row pitches, the fewer
// calls are needed.
//
// Note that this *not* check for  dst_pitch == wid
// (which could be done in a single fill much more efficiently).
//


// r0 ->  dstp
// r1 -> fillval
// r2 -> width
// r3 -> height
// r4 -> dst_stride
//
#define dstp         r0
#define fillval      r1
#define width        r2
#define height       r3
#define dst_stride   r4
#define alignlog  r6
#define itercount r8


#define iter_dest r20
#define iter_fillval r21
#define iter_wid r22
#define iter_height r23
#define iter_stride r24
#define keep_stride r25
#define iter_counter r26
#define iter_dec_at  r27

	.globl	vmemset_32_2d_general_asm
	.balign 16
	.type	vmemset_32_2d_general_asm,@function
vmemset_32_2d_general_asm:
////
// if ht <= 1, or if row pitch is multiples of VECN, just go do
//  the regular vmemset.
// @@ could also check width=1, which could be just a scalar store loop.
//
   {
      r7= fp;                                   // stack frame is speculative...
      memd(sp+#-40)=r21:20                      // since we might just jump to the other function.
      alignlog = ct0(dst_stride);
      allocframe(#32)
   } {
      p1 = cmp.gt( height,#1);                  // false if only one row
      p1 = !cmp.gt( alignlog,#VECLOGN-1 );      // false if aligned
      if( !p1.new ) jump:nt .L_jjtoit           // jump if either condition false
      if( p1.new) memd(sp+#8)=r23:22
   } {
      itercount = #1;
      alignlog = sub( #VECLOGN, alignlog );     // will be >= 1
      memd(sp+#16)=r25:24
   }
   // K = (1<<alignlog) is the number of loops we need.
   // at least 2; at most VECN.
   // (if ht is less, we only do that many, each will be 1 row).
   // Note that e.g. if K=4 and ht = 33,
   // Four calls are done, and the 'ht' values to the calls will be
   //    9, 8, 8, 8
   // this is done by finding the first height (ht/K, rounded up)
   // and then figuring out when the height should be reduced by 1
   // A value is placed in iter_dec_at, when it matches the downcount (itercount), the
   // remaining operations will be smaller in height by 1.
   //
   {
      itercount = asl( itercount, alignlog);    // 'K'
      iter_stride = asl( dst_stride, alignlog); // dst row pitch for each iter (aligned)
      iter_dest = dstp;
      memd(sp+#24)=r27:26
   } {
      iter_dec_at = add(itercount,#-1)          // K-1
      iter_counter = min( itercount,height);    // # of loops to do.
   }
   // iter_height is  (height + K-1) >> alignlog
   // iter_dec_at is iter_counter - ( height&(K-1))
   // which is 0 when ht < K, and K-(height%K) otherwise.
   {
      iter_wid = width;					// width
      iter_height = add(height,iter_dec_at);    // height + K-1
      iter_dec_at = and(height,iter_dec_at);    // height & (K-1)
   } {
      iter_dec_at = sub(iter_counter,iter_dec_at);   // K - (ht&(K-1)) [->zero when ht < K]
      iter_fillval = fillval
      keep_stride = dst_stride
      iter_height = asr( iter_height, alignlog) // height for the 1st operation
   }
   //  iter_counter = # of loops to do (>= 1)
   // r20..r24 = values for r0..4 for next call
   // iter_dec_at = loop count at which 'iter_height' needs to be 1 less
   // keep_stride = original row stride (for bumping address between call).
   //
.balign 16
.L_rcgloop:
   {
      r1:0 = combine(iter_fillval,iter_dest)    // ptr, fillval
      r3:2 = combine(iter_height,iter_wid)      // width, height
      r4 = iter_stride		// stride
      call vmemset_32_2d_asm
   } {
      iter_counter = add(iter_counter,#-1)      // count the loop...
      iter_dest = add(iter_dest,keep_stride)    // bump dest pointer by original dest pitch
   } {
      p1 = cmp.eq(iter_counter,iter_dec_at)     // time to dec 'ht'?
      if( p1.new ) iter_height  = add(iter_height,#-1)
      p0 = cmp.gt( iter_counter,#0)
      if( p0.new ) jump:t .L_rcgloop            // loop till done
   }
   // restore registers, and done...
   {
      r21:20 = memd(sp+#0 )
      r23:22 = memd(sp+#8 )
   } {
      r25:24 = memd(sp+#16 )
      r27:26 = memd(sp+#24 )
   } {
      dealloc_return
   }
.L_jjtoit:
  /// call becomes a single call to vmemset_32_2d_asm if the
  // row pitch is aligned, or if ht <= 1.
  // undo the 'allocframe' we started...
  // and just go to  vmemset_2d_asm
  // no regs have changed except fp and sp (and r6,r7 clobbered, which is ok)
   {  sp = add(fp,#8);
      fp = r7;
      jump  vmemset_32_2d_asm
   }


 .LtmpY:
      .size	vmemset_32_2d_general_asm, .LtmpY-vmemset_32_2d_general_asm
