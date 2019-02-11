/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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
   Description
     Build histogram of 8-bit inputs

    void histogram_flat_asm(
            uint16_t * histo,           // output: nbatches*256 histogram. Vec aligned.
            uint8_t const * data,       // input: data[0..depth-1] is for first histo. Any alignment.
            int depth,                  // depth of input (must be 1..65535 -- see below)
            int nbatches,               // # of batches to process; >= 1. 
            int batch_stride );         // offset from batch to batch (when nbatches>1) Any alignment.

    void histogram_d32_asm(
            uint16_t * histo,           // output: (nbatches*4)*256 histogram. Vec aligned.
            uint8_t const * data,       // input: data for first histo. must be 128 aligned
            int depth,                  // depth of input (must be 1..65535 -- see below)
            int nbatches,               // # of batches to process; >= 1.  Each batch is 4 width units.
            int d32_stride );           // offset from each d32 to next, when depth > 32. Multiple of 128.


    Overflow may occur if depth > 65535. Since the results are accumulated in sections, and the sections are summed
    using saturated adds, the results will be saturated (rather than overflowed) if each section has <= 65535 samples;
    for 'flat' mode this is guaranteed if depth <= 524,160; for 'd32' it is guaranteed when depth <= 131040.


  Reference (does not model overflow):

    void histogram_flat_asm(uint16_t * histo,  uint8_t const * data, int depth, int nbatches,   int batch_stride )
    {
        memset( histo, 0, nbatches * 256 * sizeof(uint16_t));
        for( int i =0; i < nbatches; i ++ ){
            for( int k = 0; k < depth; k++ ) histo[i*256 + data[i*batch_stride + k]] ++;
        }
    }

    void histogram_d32_asm(uint16_t * histo,  uint8_t const * data, int depth, int nbatches,   int d32_stride )
    {
        memset( histo, 0, nbatches*4 * 256 * sizeof(uint16_t));
        for( int i =0; i < nbatches*4; i ++ ){
            uint8_t const *rpd = &data[i*32];
            for( int k = 0; k < depth; k++){
                uint8_t val = rdp[(k&31) + (k>>5)*d32_stride ];
                histo[i*256 + val ] ++;
            }
        }
    }
     
*/
#define histo_ptr                   r0     //ptr to output
#define in_ptr0                     r1     //ptr to input
#define depth                       r2     //
#define nbatches                    r3     //
#define batch_stride                r4     //

#define rconst                      r5
#define srcalign                    r6
#define  in_ptr                     r8     // inner loop load pointer
#define endpoint                    r9    // srcalign + depth
#define inner_loopcount            r10
#define histo_ptr_tmp              r11

#define rtmp                        r7
#define pf_lo                      r12
#define pf_hi                      r13
#define pf_hi_lo                   r13:12

//
// M0 contains the vector-to-vector stride (#128 for _flat, and d32_stride for _d32.
// P3 is false for 'flat', true for 'd32'
//
/* ------------------------------------------------------------------------------------------ */
            .text
            .global histogram_flat_asm
            .balign 32
            .type  histogram_flat_asm @function
histogram_flat_asm:
/* ------------------------------------------------------------------------------------------ */
   {
#if defined(HIST_WITH_L2PREF)
      pf_lo = satuh(depth)
      rconst = #1
      in_ptr = in_ptr0;
    } {
      pf_lo = combine(pf_lo.l,rconst.l)	  // (depth: 1 )
      v31:30.w = vsub(v31:30.w,v31:30.w)
      pf_hi = batch_stride
    } {
      l2fetch( in_ptr0, pf_hi_lo )
#else
      in_ptr = in_ptr0;
      v31:30.w = vsub(v31:30.w,v31:30.w)
#endif
      rconst = #128
    } {
      q1 = vcmp.eq(v0.h,v0.h)         // q1 = all 1's always
      p3 = cmp.gt(r0,r0)              // p3 = false for 'flat' case
      v29:28 = vcombine(v30,v30)
    } {
      loop1( .L_flatloop,nbatches)
      v27:26 = vcombine(v30,v30)
      v25:24 = vcombine(v30,v30)
      srcalign = and(in_ptr,#127)         // get src alignment
    } {
      v23:22 = vcombine(v30,v30)
      v21:20 = vcombine(v30,v30)
      endpoint = add(srcalign,depth);      // src align + length
    } {
      v19:18 = vcombine(v30,v30)
      v17:16 = vcombine(v30,v30)
    } {
      m0 = rconst						// make this #128
      v15:14 = vcombine(v30,v30)
      v13:12 = vcombine(v30,v30)
   }
// Outer loop for 'flat version starts here.
.L_flatloop:

    // check pointer align etc.
    // inner_loop_count = ((srcalign + depth + 127)>>7) - 2
    //                    ((srcalign + depth - 129)>>7
    { 
      inner_loopcount= add(endpoint,#-129)
      q0 = vsetq(in_ptr)
      in_ptr0 = add(in_ptr0, batch_stride );        // next batch ptr
      v5:4 = vcombine(v30,v30)
    } {
      inner_loopcount= asr(inner_loopcount,#7)
      v3:2 = vcombine(v30,v30)
      v1:0 = vcombine(v30,v30)
#if __HEXAGON_ARCH__ >= 62
    } {
      loop0( .L_histloop, inner_loopcount)
      p1 = cmp.gt( inner_loopcount, #-1)       // go to special case code if inner_loopcount < 0
      q2 = vsetq2(endpoint)                    // get end mask
      srcalign = and(in_ptr0,#127);            // srcalign for next iter
    } {
      q0 = not(q0)
      v11:10 = vcombine(v30,v30);
    }
#else
      rconst = and( endpoint,#127)
    } {
     loop0( .L_histloop, inner_loopcount)
     p0 = cmp.eq(rconst,#0)
     q2 = vsetq(endpoint)
     if(!p0.new) jump:t .L_zeroq2
    } {
     q2 = not(q2)
    }
.L_zeroq2:
    {
      p1 = cmp.gt( inner_loopcount, #-1)       // go to special case code if inner_loopcount < 0
      srcalign = and(in_ptr0,#127);            // srcalign for next iter
      q0 = not(q0)
      v11:10 = vcombine(v30,v30);
    }
#endif
    {
      v7:6 = vcombine(v30,v30)
      v9:8 = vcombine(v30,v30)
    } {
      p2 = cmp.gt(inner_loopcount,#0)
      endpoint = add(srcalign,depth);          // end point for next iter.
      if(!p1) jump:nt .L_for_onevec
    } {
      v0.tmp = vmem(in_ptr++M0)                // first (mabyte partial) vector
      vhist(q0);
      if (!p2) jump:nt .L_last_histo
   }

.balign 32
.L_histloop:
    {
      v0.tmp = vmem(in_ptr++M0)
      vhist(q1)
    }:endloop0

.balign 32
.L_last_histo:
    {
      v0.tmp = vmem(in_ptr+#0)
      vhist(q2)
      p0 = cmp.gt(nbatches,#1)
    } {
      rconst = #16
      in_ptr = in_ptr0;
#if defined(HIST_WITH_L2PREF)
      if( !p0 ) jump:nt .L_no_prefetch       // skip prefetch if last batch iteration.
    }
   l2fetch( in_ptr0, pf_hi_lo );
.L_no_prefetch:
#else
    }
#endif

   // histo is done. Do the shuffles and adds, and in the process
   // clear regs v12..v31 to be ready for the next loop.
   {   // 16 shuffles, 16 adds... pack v0..v31 to v0..15
      v1:0 = vshuff( v1,v0, rconst)
      nbatches = add(nbatches,#-1)
    } {
      v0.uh = vadd(v0.uh,v1.uh):sat
      v3:2 = vshuff( v3,v2, rconst)
    } {
      v1.uh = vadd(v2.uh,v3.uh):sat
      v3:2 = vshuff( v5,v4, rconst)
    } {
      v2.uh = vadd(v2.uh,v3.uh):sat
      v5:4 = vshuff( v7,v6, rconst)
    } {
      v3.uh = vadd(v4.uh,v5.uh):sat
      v5:4 = vshuff( v9,v8, rconst)
    } {
      v4.uh = vadd(v4.uh,v5.uh):sat
      v7:6 = vshuff( v11,v10, rconst)
    } {
      v5.uh = vadd(v6.uh,v7.uh):sat
      v7:6 = vshuff( v13,v12, rconst)
    } {
      v6.uh = vadd(v6.uh,v7.uh):sat
      v9:8 = vshuff( v15,v14, rconst)
    } {
      v7.uh = vadd(v8.uh,v9.uh):sat
      v9:8 = vshuff( v17,v16, rconst)
    } {
      v8.uh = vadd(v8.uh,v9.uh):sat
      v11:10 = vshuff( v19,v18, rconst)
      v18 = vxor(v18,v18)                // start clearing regs v18..25
    } {
      v9.uh = vadd(v10.uh,v11.uh):sat
      v19 = v18
      v11:10 = vshuff( v21,v20, rconst)
    } {
      v10.uh = vadd(v10.uh,v11.uh):sat
      v13:12 = vshuff( v23,v22, rconst)
      v20 = v18
    } {
      v11.uh = vadd(v12.uh,v13.uh):sat
      v13:12 = vshuff( v25,v24, rconst)
      v21 = v18
    } {
      v12.uh = vadd(v12.uh,v13.uh):sat
      v15:14 = vshuff( v27,v26, rconst)
      v22 = v18
    } {
      v13.uh = vadd(v14.uh,v15.uh):sat
      v15:14 = vshuff( v29,v28, rconst)
      v23 = v18
    } {
      v14.uh = vadd(v14.uh,v15.uh):sat
      v17:16 = vshuff( v31,v30, rconst)
      rconst = #32
      v24 = v18
    } {
      v15.uh = vadd(v16.uh,v17.uh) :sat
      v25 = v18
      // for d32 mode, this is all the adds we need, but we still
      // need to shuffle things to get the right ordering.
      //////////////////////////////////////////////////
      // 8 shuffles, 8 adds...  pack v0..v15 to v0..v7
      // continue clearing v26..v31 and v12..v17
      v1:0 = vshuff( v1,v0, rconst)
      if( p3 ) jump .L_pack_for_d32
    } {
      v0.uh = vadd(v0.uh,v1.uh):sat
      v3:2 = vshuff( v3,v2, rconst)
      v26 = v18
    } {
      v1.uh = vadd(v2.uh,v3.uh):sat
      v3:2 = vshuff( v5,v4, rconst)
      v27 = v18
    } {
      v2.uh = vadd(v2.uh,v3.uh):sat
      v5:4 = vshuff( v7,v6, rconst)
      v28 = v18
    } {
      v3.uh = vadd(v4.uh,v5.uh):sat
      v5:4 = vshuff( v9,v8, rconst)
      v29 = v18
    } {
      v4.uh = vadd(v4.uh,v5.uh):sat
      v7:6 = vshuff( v11,v10, rconst)
      v30 = v18
    } {
      v5.uh = vadd(v6.uh,v7.uh):sat
      v7:6 = vshuff( v13,v12, rconst)
      v31 = v18
    } {
      v6.uh = vadd(v6.uh,v7.uh):sat
      v9:8 = vshuff( v15,v14, rconst)
      v17 = v18
      rconst = #64
    } {
      v7.uh = vadd(v8.uh,v9.uh):sat
     //////////////////////////////////////////////////
      // 4 shuffles, 4 adds... pack v0..v7 to v0..v3
      v1:0 = vshuff( v1,v0, rconst)
      v16 = v18
    } {
      v0.uh = vadd(v0.uh,v1.uh):sat
      v3:2 = vshuff( v3,v2, rconst)
      v15 = v18
      vmem(histo_ptr++#1)= v0.new
    } {
      v1.uh = vadd(v2.uh,v3.uh):sat
      v3:2 = vshuff( v5,v4, rconst)
      v14 = v18
      vmem(histo_ptr++#1)= v1.new
    } {
      v2.uh = vadd(v2.uh,v3.uh):sat
      v5:4 = vshuff( v7,v6, rconst)
      v13 = v18
      vmem(histo_ptr++#1)= v2.new
    } {
      v3.uh = vadd(v4.uh,v5.uh):sat
      vmem(histo_ptr++#1)= v3.new
      v12 = v18
    }:endloop1
    {
      jumpr r31        // return from 'flat' case
    }

///// for 'flat' case when everything is in one vector
.L_for_onevec:
    {
      q2 = and(q2,q0);    // get single-word mask
      jump .L_last_histo
    }
///////////////////////////////////
// continue packing for the d32 case
///////////////////////////////////

.L_pack_for_d32:
    // rconst = 32, we have histograms in v0..v15, each reg contains 4x16 sums belonging
    // to the 4 output histos. So we need to do a 16x16 transpose to the four results.
    // v1:v0 have already been transposed in-place; and v18..v25 are cleared.
    // start by transposing all the other pairs in-place, while clearing v26..v31
    {
      v15:14 = vshuff( v15,v14, rconst)
      v27:26 = vcombine(v18,v18)
      //p0 = cmp.gt(nbatches,#0)                     // any batches left after this? (already set)
    } {
      v13:12 = vshuff( v13,v12, rconst)
      v29:28 = vcombine(v18,v18)
      in_ptr0 = add( in_ptr0 ,#128)                 // move to next vector...
    } {
      v11:10 = vshuff( v11,v10, rconst)
      v31:30 = vcombine(v18,v18)                    // v18..v31 are cleared now
    } {
      v9:8 = vshuff( v9,v8, rconst)
      histo_ptr_tmp = add(histo_ptr,#(4*128))       // ->[1][0] to address first 8 vecs
    } {
      v7:6 = vshuff( v7,v6, rconst)
      histo_ptr = add(histo_ptr_tmp,#(8*128))       // ->[3][0] to address second 8 vecs
    } {
      loop0( .L_histloop, inner_loopcount)          // restore inner loop count
      v5:4 = vshuff( v5,v4, rconst)
    } {
      v3:2 = vshuff( v3,v2, rconst)                 // now we have 2x32 in each reg.
      rconst = #64
    } {
      v17:16 = vshuff( v15,v13, rconst);            // outputs [1][3] and [3][3]
      vmem(histo_ptr+#3) = v17.new                  // [3][3]
    } {
      v15:14 = vshuff( v14,v12, rconst);            // outputs  [0][3] and [2][3]
      vmem(histo_ptr+#-1) = v15.new                 // [2][3]
    } {
      v17:16 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#3) = v16                  // [1][3]
    } {
      v15:14 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#-1) = v14                 // [0][3]
    } {
      v13:12 = vshuff( v11,v9,rconst)               // [1][2] and [3][2]
      vmem(histo_ptr+#2) = v13.new                  // [3][2]
    } {
      v11:10 = vshuff( v10,v8, rconst);             // [0][2] and [2][2]
      vmem(histo_ptr+#-2) = v11.new                 // [2][2]
    } {
      v13:12 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#2) = v12                  // [1][2]
    } {
      v11:10 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#-2) = v10                 // [0][2]
    } {
      v9:8 = vshuff( v7,v5,rconst)                  // [1][1] and [3][1]
      vmem(histo_ptr+#1) = v9.new                   // [3][1]
    } {
      v7:6 = vshuff(  v6,v4, rconst);               // [0][1] and [2][1]
      vmem(histo_ptr+#-3) = v7.new                  // [2][1]
    } {
      v9:8 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#1) = v8                   // [1][1]
    } {
      v7:6 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#-3) = v6                  // [0][1]
    } {
      v5:4 = vshuff( v3,v1,rconst)                  // [1][0] and [3][0]
      vmem(histo_ptr+#0) = v5.new                   // [3][0]
    } {
      v3:2 = vshuff(  v2,v0, rconst);               // [0][0] and [2][0]
      vmem(histo_ptr+#-4) = v3.new                  // [2][0]
    } {
       v5:4 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#0) = v4                   // [1][0]
      histo_ptr = add(histo_ptr, #(4*128))
    } {
      v3:2 = vcombine(v31,v31)
      vmem(histo_ptr_tmp+#-4) = v2                  // [0][0]
      if(!p0) jumpr r31						        // done if no more batches
    } {
      v1:0 = vcombine(v31,v31)
      if(p2) jump:nt .L_last_histo;			   // just one d32 slice
      jump .L_histloop;
    }
.L_end:
      .size histogram_flat_asm, .L_end-histogram_flat_asm

/////////////////////////////////////////////////////////////////
// Entry point for histogram_d32_asm
/////////////////////////////////////////////////////////////////
            .text
            .global histogram_d32_asm
            .balign 32
            .type  histogram_d32_asm @function
histogram_d32_asm:
    {
      rconst = #128
      inner_loopcount = add(depth,#31);          // starting (depth+31)>>5
#if defined(HIST_WITH_L2PREF)
      pf_hi = batch_stride                       // this is d32 stride
#endif
      depth = and(depth,#31)
    } {
      p3 = cmp.eq(depth,#0)                          // is a multiple of 32?
      inner_loopcount = lsr(inner_loopcount,#5)      // # of d32 slices
      v31:30.w = vsub(v31:30.w,v31:30.w)
    } {
#if defined(HIST_WITH_L2PREF)
      pf_lo = combine(rconst.l,inner_loopcount.l)	 // (128: nd32 )
#endif
      inner_loopcount = add(inner_loopcount,#-1)     // now it's the proper loopcount
      rtmp = add(PC,##const_Count32@PCREL)
    } {
      if(p3) depth = #32
      in_ptr = in_ptr0;
#if defined(HIST_WITH_L2PREF)
      l2fetch( in_ptr0, pf_hi_lo )
#endif
    } {
      depth = vsplatb(depth)
      v29:28 = vcombine(v30,v30)
    } {
      v0 = vmem(rtmp+#0)                           // get vector [0,1, .. 31, 0,1... 31,  0...31, 0...31 ]
      v1 = vsplat(depth)
      v27:26 = vcombine(v30,v30)
    } {
      p2 = !cmp.gt( inner_loopcount, #0)
      q2 = vcmp.gt(v1.ub, v0.ub)                     // last-slice conditional mask (1..32 true in each quadrant)
      q1 = vcmp.eq(v0.h,v0.h)                        // q1 = all 1's always
    } {
      v25:24 = vcombine(v30,v30)
      p3 = cmp.eq(r0,r0)                             // p3 = true for 'd32' case
      m0 = batch_stride
    } {
      loop0( .L_histloop, inner_loopcount)
      v23:22 = vcombine(v30,v30)
      v21:20 = vcombine(v30,v30)
    } {
      v19:18 = vcombine(v30,v30)
      v17:16 = vcombine(v30,v30)
    } {
      v15:14 = vcombine(v30,v30)
      v13:12 = vcombine(v30,v30)
    } {
      v11:10 = vcombine(v30,v30)
      v9:8 = vcombine(v30,v30)
    } {
      v7:6 = vcombine(v30,v30)
      v5:4 = vcombine(v30,v30)
      in_ptr0 = add(in_ptr0, #128)             // address for next outer loop
    } {
      v3:2 = vcombine(v30,v30)
      v1:0 = vcombine(v30,v30)
      if(p2) jump:nt .L_last_histo;            // just one d32 slice
      jump .L_histloop;
   }
.L_endd32:
      .size histogram_d32_asm, .L_endd32-histogram_d32_asm
