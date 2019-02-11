
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
	.file	"vmemcpy_2d_h.S"

//
// HVX rectangle copy operation
//
// void vmemcpy_2d_asm(
//		unsigned wid,		// must be > 0; bytes
//      unsigned ht,		// must be > 0; rows
//      void *dst,			// destination address, any allowed
//      int dst_pitch,		// row pitch of dest; must be multiple of vector
//      void const *src,	// source address, any allowed
//      int src_pitch)		// row pitch of source; must be multiple of vector
//
// This operation does any rectangle copy using vector operations; it uses masked
// writes as needed to avoid over-writing portions of the output buffer outside the
// specified area.
// The row pitches must be vector aligned (so whatever strategy is used on one row,
// is used on all others). When the width is smallish - one or two dest vectors per row -
// there is no 'horizontal' loop, just the height loop.
//
// All reads & writes are done with vector ops; it will only read source vectors which
// contain bytes that need to be copied to the output.
//
// The input and output regions must be disjoint (though, the address ranges
// may overlap. e.g. copying a tile horizontally within one image buffer is fine).
//
// The row-pitch condition is not checked. The wid >0, ht >0 condition is checked, and
// the routine does nothing if this is not the case.
//
// This will be a little faster for __HEXAGON_ARCH__ >= 62
// (uses vsetq2, and conditional vector load)
//
//
// There is also vmemcpy_2d_general_asm
// .. which has exactly the same form but no restriction on src/dst pitch.
// It will make multiple calls to vmemcpy_2d_asm as needed; for instance
// if the src/dst pitch are a multiple of 1/4 vector (but not 1/2) it will
// make 4 calls, each of which copies every 4th row.
//

#ifndef __HVXDBL__
#define VECN 64
#define VECLOGN 6
#else
#define VECN 128
#define VECLOGN 7
#endif
// r0 ->  wid
// r1 -> ht
// r2 -> dstp
// r3 -> dst_pitch
// r4 -> srcp
// r5 -> src_pitch
//
//
	.globl	vmemcpy_2d_asm
	.balign 16
	.type	vmemcpy_2d_asm,@function
vmemcpy_2d_asm:
  {
	r6 = sub(r4,r2);		// src-dst: number for valign
	r7 = and(r2,#VECN-1);		// destination offset
	r10 = and(r4,#VECN-1);	// source offset
	q0 = vsetq(r2);			// the 'left' mask (needs ~ though)
  }
  // set q0 up to be  0 0 0 .. 1 1 1
  // to be the 'start' mask for the first output vector in each row;
  //  and q1 to  1 1 1 .. 0 0 0 to be the 'end' mask for the last vector.
  // If there is only one output vector per row we'll and them later.
  {
  		r15 = min(r0,r1);	// protect against ht <=0, wid <= 0
  		r2 = sub(r2,r7);		// align the dest address.
  		r8 = add(r7,r0);		// if this > VECN, needs >1 vector write per row
		r6 = and(r6,#VECN-1);
  }
  {
  		p0 = !cmp.gt(r15,#0);		//protect against ht,wid <= 0
  		if(p0.new) jump:nt .L_alldone;		// done if ht <= 0 or wid <= 0
#if __HEXAGON_ARCH__ < 62
  		r9 = and(r8,#VECN-1);		// test if end falls on a boundary
  		q1  = or(q0,!q0);		// the 'end' mask (last partial write - all 1's if falls on a boundary).
  }
  {    p1 = cmp.eq(r9,#0);	//
  	    q0 = not(q0);		// correct the 'start' mask
  		if( p1.new) jump:nt .L_mskskip;

  }
  {
		q1 = vsetq(r8);		// this the 'end' mask in the general case.
  }
.L_mskskip:
#else
  	    q0 = not(q0);		// correct the 'start' mask
		q1 = vsetq2(r8);
 }
#endif
  {
  	// test if > 1 output vector per row is needed.
  		r8 = add(r8,#-(VECN+1));		// for later...
  	    P0=!cmp.gt(r8,#VECN);      //
  	    if( P0.new) jump:nt .L_short_rows;		// r8 >= 0 if not taken.
  }
//
// OK, we are writing at least two output vectors, which means there is at least a 'first' vector (using q0)
// and zero or more 'middle' vectors, and a 'last' vector (using q1).
// - Either or both of q0,q1 could be all 1's (but last of q0, and first of q1, are always 1).
// - the number of vectors read could be 1 less, same as, or one more, than the number written.
// We take a special case for src and dest having the same alignment, which is simpler since there's no rotate
// and the # read is always the same as the number written.
// First step: find the number of 'middle' vectors, and also check that aligment (r6=0)

// note: lower  bits of r8 are the offset of the last dst byte in vector
// by adding r6, we get the offset of the last src byte in vector.
//
	{
		r11 = lsr(r8,#VECLOGN)		// this is the # of 'middle' vecs   (>=0)
		r15 = add(r8,r6);			// lower bits are the offset of last byte in src vector
		p0 = cmp.eq(r6,#0);			// do both have the same alignment?
		if( !p0.new) jump:t .L_general_unaligned;
	}
	//
	// the general aligned case
	//
	{
		p1 = cmp.gt(r11,#0);			// true if any 'middle' vectors to deal with
		if( !p1.new) jump:t .L_ga_aligned_two;
	  	r4 = sub(r4,r7);				// align the src address.
		loop1(.L_ga_outer_loop,r1);		// loop per row
	}
.L_ga_outer_loop:
	{
		r14 = add(r4,r5);		// find next row source address
		r15 = add(r2,r3);		// find next row dest address
		v0 = vmem(r4++#1);		// load first vector
		loop0(.L_ga_inner_loop,r11);
	}
	{
		if( q0 ) vmem(r2++#1)= v0;  // first vector store
		v0= vmem(r4++#1);		// load next...
	}
.falign
.L_ga_inner_loop:
	{
		vmem(r2++#1) = v0;			// just copy...
		v0= vmem(r4++#1);
	}:endloop0;
	{
		if( q1 ) vmem(r2+#0)= v0;  // last vector store
		r2 = r15;					// move addresses to next row.
		r4 = r14;
	}:endloop1;
.L_alldone:
	{ jumpr r31; }					// !! all done

/// aligned, two per row.
.L_ga_aligned_two:

	{
		loop0(.L_ga_outer_loop_aligned_2,r1);		// loop per row
	}
.falign

.L_ga_outer_loop_aligned_2:
	{
		v0.cur = vmem(r4+#0);		// load first vector
		if( q0 ) vmem(r2+#0)= v0;  // first vector store
	}
	{
		v0.cur = vmem(r4+#1);		// load first vector
		if( q1 ) vmem(r2+#1)= v0;  // last vector store
		r2 = add(r2,r3);		// find next row dest address
		r4 = add(r4,r5);		// find next row source address
	}:endloop0;
	{ jumpr r31; }					// !! all done

//  this is the general unaligned case. Each row will consist of:
//  (1) optional preload of an 'extra' vector (if P2 is true)
//  (2) load of the following vector (first if !P2)
//  (3) valign of those two gives the vector for the first store (masked by q0)
//    repeat r11 times:
//     load another vector, valign to get a vector which is stored directly
//  (4) *maybe* load a final vector (if P3 is true)
//  (5)   another valign to get a value for the final store (masked by q1).
// So we need to calculate P2,P3 and set up the source pointer properly now.
//
// note:  Soff,Doff are the source/dest offsets
//      Send = (Soff+wid-1)%VECN is the offset of the last source byte
//      Dend = (Doff+wid-1)%VECN is the offset of the last dest byte
//
//        Ncore is the number of 'core' vec writes (in r11,>=0)
//        p2 = 1 (if Soff>=Doff) else 0		(in this situation Soff != Doff)
//        p3 = 1 if( Send <Dend )else 0

//
.L_general_unaligned:
	{
		r15 = and(r15, #VECN-1);	// offset of last source byte
		r8 = and(r8, #VECN-1);		// offset of last dest byte
		p1 = cmp.gt(r11,#0);			// true if any 'middle' vectors to deal with
		p2 = !cmp.gt(r7,r10);			// if src_off >= dst_off, we need extra initial load
	}
	{
		r4 = sub(r4,r10);				// align source (first vec containing needed bytes)
		P3 = cmp.gt(r8,r15);			// p3 true if dest-last-bvyte  > src_last_byte
		m0 = r5;
	}
	{
		loop1(.L_gb_outer_loop, r1);		// set up loop
		// if 'p2' we actually start each row with r4 pointing at the
		// second vector, and pick up the first one conditionally.
		if(p2) r4 = add(r4 ,#VECN);
		if(!p1) jump:nt .L_gb0_cases;
	}
.falign
.L_gb_outer_loop:
	{
		r14 = add(r4,r5);		// find next row source address
		r15 = add(r2,r3);		// find next row dest address
		 v0 = vmem(r4++#1);		// load first vector
#if __HEXAGON_ARCH__ < 62
		 if(!p2) jump:nt .L_skipv1
	}
	{    v1 = vmem(r4+#-2);		// load extra 'left' vector
	}
.L_skipv1:
#else
	}
#endif
	{
#if __HEXAGON_ARCH__ >= 62
	    if(p2)v1.cur = vmem(r4+#-2);		// load extra 'left' vector
#endif
		// if p2 is false, the content of v1 is indeterminate here, but the
		// indeterminate result lanes of v2 will not be stored.
		v2 = valign( v0, v1, r6);	//  //first data to write
	}
	{
		if(q0) vmem(r2++#1)= v2;		// write first data
		loop0(.L_gb_inner_loop,r11);
	}
.falign
.L_gb_inner_loop:
	 {
		v1 = vmem(r4++#1);		// next vector
	}
	{
 		v2 = valign( v1, v0, r6);	// align ... first data to write
		v0 = v1;
 	vmem(r2++#1) = v2.new;
  }:endloop0;
#if __HEXAGON_ARCH__ < 62
	{ if(!P3) jump:nt .L_skiplast;
	}
	{
		 v1  =vmem(r4);		// get last source word
	}
.L_skiplast:
	{
#else
	{
	   if(p3) v1.cur = vmem(r4+#0);	// get last source word
#endif
	   v2 = valign(v1,v0,r6);		// set up last operation
	}
	{
		if(q1) vmem(r2) = v2;		// write last
		r2 = r15;			// next row...
		r4 = r14;
	}:endloop1;

	{ jumpr r31; }					// !! all done

// best inner loop for the unaligned case:
//  {	vmem(r2++#1) = v2.new;	v2 = valign( v1, v0, r6);  v1.cur = vmem(r4++#1);  }
//  {	vmem(r2++#1) = v2.new;	v2 = valign( v0, v1, r6);  v0.cur = vmem(r4++#1);  }

//
// special case. unaligned, when the inner loop count is 0
// I.e. we are writing 2 output vecs.
// there are four subcases:
// A  p2 = 0  p2 = 0		# read 1 per loop; vror; write the same value twice.
// B p2 = 0, p3 = 1		# read 2 per loop align { v0, x} and { v1, v0}
// C p2 = 1  p3 = 0		# read 2 per loop align { v1,v0} and { x, v1}
// D p2 = 1  p3 = 1		# read 3 per loop; align {v1,v0} and {v2,v1}
//
.L_gb0_cases:
	{ if (p2) jump:nt .L_gb0_caseCD ; if (p3) jump:t .L_gb0_caseB }

//// no middle vecs, p2=p3 = 0
// In this case, the source bytes for each row are in
// one vector; just need to vror it and  write it twice.
.L_gb0_caseA:
	{loop0(.L_gb0A_outer_loop, r1);		// set up loop
	}
.falign
.L_gb0A_outer_loop:
	{
		 v1.cur = vmem(r4++m0);		// load first vector (only vec)
		v2 = vror( v1, r6);	//  data to write
	}
	{
		if(q0) vmem(r2+#0)= v2;		// write first data
	}
	{
		if(q1) vmem(r2+#1) = v2;		// write last
		r2 = add(r2,r3);		// find next row dest address
	}:endloop0;

	{ jumpr r31; }					// !! all done

//// no middle vecs, p2=0,p3 =1
.L_gb0_caseB:
	{loop0(.L_gb0B_outer_loop, r1);		// set up loop
	r4 = add(r4,#VECN);				// point to second input vector.
	}
.falign
.L_gb0B_outer_loop:
	{
		 v0.cur = vmem(r4+#-1);		// load first vector
		v2 = vror( v0, r6);	//  //first data to write
	}
	{
		v1.cur = vmem(r4++m0)
		if(q0) vmem(r2+#0)= v2;		// write first data
		v0 = valign(v1,v0,r6);		// last operation
	}
	{
		if(q1) vmem(r2+#1) = v0;		// write last
		r2 = add(r2,r3);		// find next row dest address
	}:endloop0;

	{ jumpr r31; }					// !! all done


.L_gb0_caseCD: { if (p3) jump:nt .L_gb0_caseD }
//// no middle vecs, p2 = 1, p3 = 0;
.L_gb0_caseC:
	{loop0(.L_gb0C_outer_loop, r1);		// set up loop
	}
.falign

.L_gb0C_outer_loop:
	{
		 v0 = vmem(r4+#-1);		// load first vector
	}
	{
		v2 = valign( v1,v0, r6);	//  //first data to write
		v1.cur = vmem(r4++m0);			// second vector
	}
	{
		if(q0) vmem(r2+#0)= v2;		// write first data
		v0 = vror(v1,r6);		// last operation
	}
	{
		if(q1) vmem(r2+#1) = v0;		// write last
		r2 = add(r2,r3);		// find next row dest address
	}:endloop0;

	{ jumpr r31; }					// !! all done


//// no middle vecs, p2=1,p3 =1
/// (i.e. 3 in, 2 out per row)

.L_gb0_caseD:
	{loop0(.L_gb0D_outer_loop, r1);		// set up loop
	r4 = add(r4,#VECN);				// point to last input vector.
	}
.falign
.L_gb0D_outer_loop:
	{
		 v0 = vmem(r4+#-2);		// load first vector of 3
	}

	{    v1.cur = vmem(r4+#-1);		// load second vector
		v2 = valign( v1, v0, r6);	//  //first data to write
	}
	{
		if(q0) vmem(r2+#0)= v2;		// write first data
		 v0.cur  =vmem(r4++m0);		// get last source word
		v2 = valign(v0,v1,r6);		// last operation
	}
	{
		if(q1) vmem(r2+#1) = v2;		// write last
		r2 = add(r2,r3);		// find next row dest address
	}:endloop0;

	{ jumpr r31; }					// !! all done

////////////////////////////////////////////////////////////////////////////////////////////
// cases where only one vector needs to be written per row
// (columns do not span a vector boundary in the output)
//
.falign
.L_short_rows:	// only one output vector per row...
  	// test if we need two reads or just one per row.
  	{
  		r11 = add(r10,r0);		// if > VECN, need more than one read per row
  		q0 = and(q1,q0);		// generate the proper mask for the write
		m1 = r3;				// dest row pitch
	}
	{
  	    P0=!cmp.gt(r11,#VECN);      //
  	    r8 = sub(r4,r10);			// align the source
  	    if( P0.new) jump:nt .L_short_read;
	}
	// ok, we need two reads per row;  can just do an unaligned read.
	// subtract the dest offset from the source address (note that the
	// source offset must be > the dest, so this will not cross a boundary)
	//
	{
		loop0(.L_short_short_loop,r1);
		r4 = sub(r4,r7);			// source address for unaligned reads
	}
.falign
.L_short_short_loop:
	{
		v0 = vmemu(r4);			// get a value
		r4 = add(r4,r5);
	}
	{
		if( q0 ) vmem(r2++m1) = v0;
	}:endloop0;
	{ jumpr r31; }					// !! all done

.L_short_read:
// this is the case where only need one read and one write per row.
// if r6 != 0, we need to vror by r6, otherwise we can just copy.
// Also, if == 1 rows, treat as the vror case (so I don't need
// to handle both degenerate loop cases).
//
	{ P0 = cmp.eq(r6,#0);
	  P1 = cmp.gt(r1,#1);
	  m0 = r5;
	 }
	 {
	  v0 = vmem(r8++m0);		// get the first vector
	  p0 = and(p0,p1);
		if(!p0.new) jump:nt .L_short_short_unaligned;
	 r1 = add(r1,#-1);
	}
// aligned copy, at least 2 rows
	{
		loop0(.L_short_2_loop,r1);
	}
.falign
.L_short_2_loop:
	{
		if(q0) vmem(r2++m1) = v0;
		v0 = vmem(r8++m0);
	}:endloop0;
	{if(q0) vmem(r2+#0) = v0;
	 jumpr r31; }					// !! all done
//
// case we only need to read one vector per row,
// but we need to do a ror. The case where ht ==1
// also gets here.
//
.falign
.L_short_short_unaligned:
	{ v2 = vror(v0,r6);
	  if(!p1) jump:nt .L_justone;
		loop0(.L_short_3_loop,r1);
	 }
.falign
.L_short_3_loop:
	{
		if(q0) vmem(r2++m1) = v2;
		v0.cur = vmem(r8++m0);
		v2 = vror(v0,r6);
	}:endloop0
 .L_justone:
	{if(q0) vmem(r2+#0) = v2;
	 jumpr r31; }					// !! all done


.LtmpX:
	.size	vmemcpy_2d_asm, .LtmpX-vmemcpy_2d_asm


////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
// General copy, which allows any rowpitch,
// by calling the other copy multiple times as needed.
// The better the alignment of the supplied row pitches, the fewer
// calls are needed.
//
// Note that this *not* check for src_pitch == dst_pitch == wid
// (which could be done in a single copy much more efficiently).
//

	.globl	vmemcpy_2d_general_asm
	.balign 16
	.type	vmemcpy_2d_general_asm,@function
vmemcpy_2d_general_asm:

// r0 ->  wid
// r1 -> ht
// r2 -> dstp
// r3 -> dst_pitch
// r4 -> srcp
// r5 -> src_pitch

//
// if ht <= 1, or of both row pitches are multiples of VECN, just go do
//  the regular copy.
//
	{
		r6 = or(r3,r5)
		r7= fp;					// stack frame is speculative...
		memd(sp+#-48)=r21:20	// since we might just jump to the other function.
		allocframe(#40)
	}
	{
		p0 = cmp.gt( r1,#1);
		r6 = ct0(r6);
		r8 = #1;
		memd(sp+#8)=r23:22
	}
	{

		p1 = !cmp.gt( r6,#VECLOGN-1 );	// false if aligned
		p1 = p0;						// false if only 1 row
		if( !p1.new ) jump:nt .L_jjtoit;	// jump if either condition false
		r6 = sub( #VECLOGN, r6 );		// will be >= 1
	}
	// K = (1<<r6) is the number of loops we need.
	// at least 2; at most VECN.
	// (if ht is less, we only do that many, each will be 1 row).
	// Note that e.g. if K=4 and ht = 33,
	// Four calls are done, and the 'ht' values to the calls will be
	//    9, 8, 8, 8
	// this is done by finding the first height (ht/K, rounded up)
	// and then figuring out when the height should be reduced by 1
	// A value is placed in r27, when it matches the downcount (r26), the
	// remaining copies will be smaller by 1.
	//
	{
		r8 = asl( r8, r6);			// 'K'
		r23  =asl( r3, r6);			// dst row pitch for each op
		r22 = r2;
		memd(sp+#16)=r25:24
	}
	{
		r25  =asl( r5, r6);			// src row pitch for each op
		r24 = r4;					// src  pointer
		r27 = add(r8,#-1)			// K-1
		memd(sp+#24)=r27:26
	}
	{
		r26 = min( r8,r1);			// # of loops to do.
		r20 = r0;					// width
		r21 = add(r1,r27);			// height + K-1
		r27 = and(r1,r27);			// height & (K-1)
	}
	{
		r27 = sub(r26,r27);			// K - (ht&(K-1)) [->zero when ht < K]
		memd(sp+#32) = r19:18
		r19:18 = combine(r5,r3)		// save row pitches
		r21 = asr( r21, r6)			// height for the 1st operation
	}
	//  r26 = # of loops to do (>= 1)
	// r20..r25 = values for r0..5 for next call
	// r27 = loop count at which 'height' (r21) needs to be 1 less
	// r19,18 = original row pitches (for bumping address between call).
	//
.L_rcgloop:
	{
		r1:0 = combine(r21,r20)		// width, height
		r3:2 = combine(r23,r22)		// dst addr and pitch
		r5:4 = combine(r25,r24)		// src addr/pitch
		call vmemcpy_2d_asm
	}
	{
		r26 = add(r26,#-1);		// count the loop...
		r22 = add(r22,r18);		// bump dest pointer by original dest pitch
		r24 = add(r24,r19)		// and src address too
	}
	{
		p1 = cmp.eq(r26,r27)		// time to dec 'ht'?
		if( p1.new ) r21  = add(r21,#-1)
		p0 = cmp.gt( r26,#0)
		if( p0.new ) jump:t .L_rcgloop;	// loop till done
	}
	// restore registers, and done...
	{
		r21:20 = memd(sp+#0 )
		r23:22 = memd(sp+#8 )
	}
	{
		r25:24 = memd(sp+#16 )
		r27:26 = memd(sp+#24 )
	}
	{
		r19:18 = memd(sp+#32 )
		dealloc_return
	}
.L_jjtoit:
  /// call becomes a single call to hvx_rectangle_copy_FUNCNAME if the
  // row pitches are aligned, or if ht <= 1.
  // undo the 'allocframe' we started...
  // and just go to  hvx_rectangle_copy
  // no regs have changed except fp and sp.
   { 	sp = add(fp,#8);
  	 	fp = r7;
        jump  vmemcpy_2d_asm
    }


 .LtmpY:
	.size	vmemcpy_2d_general_asm, .LtmpY-vmemcpy_2d_general_asm
