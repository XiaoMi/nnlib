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
   Desrciption
     Perform 1d integral over activation inputs of size in_depth. The in_depth blocks are 
     collapsed to 1 int.
   Requirements
     Input must be aligned to 128 bytes, data can be non-multiple of 128bytes. Final store will
     write only to amount needed. If multithreaded the data mut be split into multiples of 32 inputs
     with final call having non -multiple of 32. E.g 35*35 into 4 threads. 
     3 * 320 + 1 * 265
   Codesize
     256 bytes, no stack
 */
#if 0
void gvsum_cn(uint8_t * in_data, int * out_sum, int in_width, int in_depth,  int out_height,
              int stride_vert, int filt_offset)
{
  int i,j,k,l,sum;

  for(j=0; j < out_height; j+=1) {
      for(i=0; i < in_width; i++) {
          sum = 0;
          for(l = 0; l < in_depth/32; l++) {
              for(k=0; k < 32; k++) 
                  sum -= filt_offset*in_data[in_depth*in_width*stride_vert*j+32*in_width*l+32*i+k];
          }
          *out_sum++ = sum;
      }
  }
}

/* ------------------------------------------------------------------------------------------ */
            .text
            .global gsum_asm
            .balign 32
            .type  gsum_asm, @function
gsum_asm:
/* ------------------------------------------------------------------------------------------ */
#define in_ptr                      r0     //ptr to activation bytes
#define out_ptr                     r1     //ptr to output int sums
#define width                       r2     //in == out width of image logical % 4
#define depth                       r3     //indepth multiple of 32  
#define out_height                  r4     //number of logical output lines
#define stride_vert                 r5     //vertical stride
#define filt_offset                 r6     //offset for weights
#define PV(VSRC) .word (0x1DFFE020+VSRC)   //debug print vector reg
/* ------------------------------------------------------------------------------------------ */
#define c32                         r9     //const 32
#define c16                         r8     //const 16
#define c32c16                      r9:8   //paked consts
#define cm4                         r7     //-4 for dealing out ints
#define width_depth                 r5     //width * depth
#define in_ptr0                     r10    //temp input ptr
#define in_ptr1                     r4     //temp input ptr
#define col_count                   r12    //count out width
#define store_count                 r13    //store every 8 or 10 iterations
#define store_col_count             r13:12 //apcked counters
#define next_d32_row                r13    //width of sum image in bytes
#define area                        r4     //total number of computations
#define cm1cm4                      r15:14 //packet constants
#define total_count                 r16    //num bytes left to store
#define   mdst     r4
#define   dalign    r12
/* ------------------------------------------------------------------------------------------ */
#define qprolog                     q1               
#define qepilog                     q2               
#define d3d2d1d0                    v11    //data from activations
#define a8a8a8a8                    v12    //packed sum of activations 4 to 1
#define s8s8s8s8                    v13    //sum of activations across depths 32
#define s8s8s8s8h_s8s8s8s8l         v17:16 //dealt activations odd and even
#define s8s8s8s8h                   v17    //dealt activations odd 
#define s8s8s8s8l                   v16    //dealt activations even
#define acc31_00                    v10    //fifo containing history of sums of depths
#define aout                        v14    //temp output
#define zero                        v15    //zero out reg
/* --------------------------------------------------------------------------------------- */
     { 
       allocframe(#16)
       filt_offset = memw(sp+#0<<2)                       //filter offset
     } {
       memd(sp+#0) = r17:16                               //
       cm4 = #-4                                          //constant -4
       p1 = and(p1, !p1)                                  //set p1 to 0
       next_d32_row = asl(width, #5)                      //width of row in bytes
     } {
       c32c16 = combine(#32, #16)                         //setup consts
       M0 = next_d32_row                                  //set up modifier reg
     } {
       q1 = vsetq(c16)                                    //--|___
       area = mpyi(width, out_height)                     //total number of sums
       filt_offset = vsplatb(filt_offset)                 //pack into bytes
     } {
       width_depth = mpy(stride_vert.L, depth.L)          //create vertical stride
       total_count = asl(area, #2)                        //get int mask input
       r14 = #0x7f                                        //look at low 7 bits
       q0 = vsetq(c32)                                    //----|____
     } {
       p0 = bitsclr(total_count, r14)                     //are low 7 bits 0?
       width_depth = mpyi(width_depth, width)             //vertical stride
       cm1cm4 = combine(#-1, #-4)                         //pack constsn
     } {
       area = lsr(area, #2)                               //4 pts per inner loop
       depth = lsr(depth, #5)                             //how many 32byte chunks to sum
       in_ptr0 = in_ptr                                   //temp in ptr
     } {
       area = add(area, #2)                               //7 * 4 + 2 * 4
       in_ptr = add(in_ptr, width_depth)                  //update input ptr to next row
     } {
       loop1(.L_while, area)                              //number of computations
       q0 = and(q0, !q1)                                  //__|--|__ vec predicate
       store_col_count = combine(#10, width)              //p0 = 0
       p0 = and(p0, !p0)
     } {
       loop0(.L_sum, depth)                               //set up inner loop of horz sum
       in_ptr1  = in_ptr0                                 //temp ptr
       p1 = and(p1, !p1)                                  //p1 = 0
       zero = #0                                          //#0
     }
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_while:
     { a8a8a8a8 = #0                                      //clear accumulator
       in_ptr0 = add(in_ptr0, #128)                       //update to next 4 depth
       acc31_00 = vmux(q0, s8s8s8s8, acc31_00)            //merge in 4 new ints
       if(p0) in_ptr = add(in_ptr, width_depth)           //if at end of row, next row
     }
.L_sum:
     { d3d2d1d0.tmp = vmem(in_ptr1++M0)                   //load 4 new depths
       a8a8a8a8.uw += vrmpy(d3d2d1d0.ub, filt_offset.ub)  //32 sums of 4 make negative
       if(p1) store_count = #8                            //if 8 or 10 iterations reset
       if(p0) col_count = width                           //reset line count
     }:endloop0
     { store_col_count = vaddw(store_col_count, cm1cm4)   //decrement counters
       loop0(.L_sum, depth)                               //set up inner loop of horz sum
       aout= vror(acc31_00, c32)                          //rotate out of way 16bytes 
       p0 = cmp.eq(col_count, #4)                         //if col count expired
     } {
       s8s8s8s8h_s8s8s8s8l = vdeal(a8a8a8a8,s8s8s8s8,cm4) //deal in new outputs
       aout.w = vsub(zero.w, aout.w)                      //make sum -ve
       if(p0) in_ptr0 = in_ptr                            //update to next indepth*in_width 
     } {
       if(p1) vmemu(out_ptr++#1) = aout                   //write after 10 then 8 finished ints
       if(p1) total_count = add(total_count, #-128)       //
     } {
       s8s8s8s8.w = vadd(s8s8s8s8h.w, s8s8s8s8l.w)        //32sums of 8 final block into tmp  
       acc31_00 = vror(acc31_00, c16)                     //rotate out fo way 16bytes 
       in_ptr1  = in_ptr0                                 //update vertical depth  count
       p1 = cmp.eq(store_count, #0)                       //ready to store new set of sums?
     }:endloop1
     {
       acc31_00 = vmux(q0, s8s8s8s8, acc31_00)            //merge in 4 new ints
       mdst = and(out_ptr, #127)                          //ptr to mod 128
       qprolog = vsetq(out_ptr)                    //find dist to ptr
       c32 = sub(#32, total_count)                        //compute how many values left
     } {
       dalign = add(mdst, total_count)                    //dist to end of data
       mdst = sub(c32, out_ptr)                     //create left rotate
     } {
       aout = vror(acc31_00, mdst)                  //rotate left by ptr
     } {
       qepilog = vsetq(dalign)                      //do mask for 2nd store
       p1 = cmp.gt(dalign, #127)                    //is it a double store?
       if(p1.new) jump:nt .L_gt1280                 //skip over logic for 1 part store
       aout.w = vsub(zero.w, aout.w)                      //make sum negative
     } {
       qprolog = or(qprolog, !qepilog)              //compound 2 masks
       qepilog = and(qprolog, !qprolog)             //cancel 2nd mask
     }
.L_gt1280:
     { if(!qprolog) vmem(out_ptr+#0) = aout     //store 1st part, in ptr
     } {
       if( qepilog) vmem(out_ptr+#1) = aout      //cond store 2nd part
       r17:16 = memd(sp+#0)
     } {
       dealloc_return                                     //return
     }
.L_end:
/*=============================================================================*/
      .size gsum_asm, .L_end-gsum_asm
/*=============================================================================*/
#else
/* ------------------------------------------------------------------------------------------ */
            .text
            .global gsum_asm
            .balign 32
            .type  gsum_asm, @function
gsum_asm:
/* ------------------------------------------------------------------------------------------ */
#define in_ptr                      r0     //ptr to activation bytes
#define out_ptr                     r1     //ptr to output int sums
#define width                       r2     //in == out width of image logical % 4
#define depth                       r3     //indepth multiple of 32  
#define out_height                  r4     //number of logical output lines
#define stride_vert                 r5     //vertical stride
#define PV(VSRC) .word (0x1DFFE020+VSRC)   //debug print vector reg
/* ------------------------------------------------------------------------------------------ */
#define width_depth                 r5     //width * depth
#define cm4                         r6     //-4 for dealing out ints
#define cm1cm4                      r7:6   //packet constants
#define c32                         r9     //const 32
#define c16                         r8     //const 16
#define c32c16                      r9:8   //paked consts
#define in_ptr0                     r10    //temp input ptr
#define in_ptr1                     r11    //temp input ptr
#define col_count                   r12    //count out width
#define store_count                 r13    //store every 8 or 10 iterations
#define store_col_count             r13:12 //apcked counters
#define next_d32_row                r13    //width of sum image in bytes
#define area                        r14    //total number of computations
#define out_count                   r14
#define filt_offset                 r15    //offset for weights
#define offset                      r28         
/* ------------------------------------------------------------------------------------------ */
#define d3d2d1d0                    v11    //data from activations
#define a8a8a8a8                    v12    //packed sum of activations 4 to 1
#define s8s8s8s8                    v13    //sum of activations across depths 32
#define s8s8s8s8h_s8s8s8s8l         v17:16 //dealt activations odd and even
#define s8s8s8s8h                   v17    //dealt activations odd 
#define s8s8s8s8l                   v16    //dealt activations even
#define acc31_00                    v10    //fifo containing history of sums of depths
#define aout                        v14    //temp output
#define zero                        v15    //zero out reg
/* --------------------------------------------------------------------------------------- */
     { filt_offset = memw(sp+#0<<2)                       //filter offset
       c32c16 = combine(#32, #16)                         //setup consts
       next_d32_row = asl(width, #5)                      //width of row in bytes
     } {
       M0 = next_d32_row                                  //set up modifier reg
       area = mpyi(width, out_height)                     //total number of sums
       q1 = vsetq(c16)                                    //--|___
       in_ptr0 = in_ptr                                   //temp in ptr
     } {
       filt_offset = vsplatb(filt_offset)                 //pack into bytes
       width_depth = mpy(stride_vert.L, depth.L)          //create vertical stride
       q0 = vsetq(c32)                                    //----|____
       zero = #0                                          //#0
     } {
       width_depth = mpyi(width_depth, width)             //vertical stride
       cm1cm4 = combine(#-1, #-4)                         //pack constsn
       offset = extractu(width,#3,#2)                     //
       q0 = and(q0, !q1)                                  //__|--|__ vec predicate
     } {
       area = lsr(area, #2)                               //4 pts per inner loop
       depth = lsr(depth, #5)                             //how many 32byte chunks to sum
       offset = add(offset,#-1)                           //
     } {
       loop1(.L_while, area)                              //number of computations
       p0 = or(p0, !p0)                                   //
       store_count  = #10                                 //
       offset = and(offset,#7)                            //
     } {
       p2 = not(p0)                                       //
       loop0(.L_sum, depth)                               //set up inner loop of horz sum
     }
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_while:
     { if (!p2) out_count = width                         //
       in_ptr1 = in_ptr0                                  //update vertical depth  count
       in_ptr0 = add(in_ptr0, #128)                       //update to next 4 depth
       p1 = cmp.eq(store_count, #0)                       //ready to store new set of sums?
     } {
       a8a8a8a8 = #0                                      //clear accumulator
       acc31_00 = vmux(q0, s8s8s8s8, acc31_00)            //merge in 4 new ints
       if(p0) in_ptr = add(in_ptr, width_depth)           //if at end of row, next row
       if(p0) col_count = width                           //reset line count
     } 
.L_sum:
     { d3d2d1d0.tmp = vmem(in_ptr1++M0)                   //load 4 new depths
       a8a8a8a8.uw += vrmpy(d3d2d1d0.ub, filt_offset.ub)  //32 sums of 4 make negative
       nop; nop                                           //
     }:endloop0

     { store_col_count = vaddw(store_col_count, cm1cm4)   //decrement counters
       loop0(.L_sum, depth)                               //set up inner loop of horz sum
       aout= vror(acc31_00, c32)                          //rotate out of way 16bytes 
       p0 = cmp.eq(col_count, #4)                         //if col count expired
     } {
       s8s8s8s8h_s8s8s8s8l = vdeal(a8a8a8a8,s8s8s8s8,cm4) //deal in new outputs
       aout.w = vsub(zero.w, aout.w)                      //make sum -ve
       if(p1) vmem(out_ptr++#1) = aout.new                //write after 10 then 8 finished ints
       if(p0) in_ptr0 = in_ptr                            //update to next indepth*in_width 
     } {
       if (p1) store_count = #7                           //if 8 or 10 iterations reset
       if (p1) out_count = sub(out_count,c32)             //
       acc31_00 = vror(acc31_00, c16)                     //rotate out fo way 16bytes 
     } {
       p2 = cmp.gt(out_count,#0)                          //
       if (!p2.new) store_count = offset                  //
       if (!p2.new) out_height = add(out_height,#-1)      //
       s8s8s8s8.w = vadd(s8s8s8s8h.w, s8s8s8s8l.w)        //32sums of 8 final block into tmp  
     }:endloop1
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_post:
     { p1 = cmp.eq(store_count, #0)                       //ready to store new set of sums?
       if (!p2) out_count = width                         //
       acc31_00 = vmux(q0, s8s8s8s8, acc31_00)            //merge in 4 new ints
       store_count = add(store_count, #-1)                //decrement counters
     } {
       s8s8s8s8h_s8s8s8s8l = vdeal(a8a8a8a8,s8s8s8s8,cm4) //deal in new outputs
       if (p1) store_count = #7                           //if 8 or 10 iterations reset
       if (p1) out_count = sub(out_count,c32)             //
     } {
       aout= vror(acc31_00, c32)                          //rotate out of way 16bytes 
       p2 = cmp.gt(out_count,#0)                          //
       if (!p2.new) store_count = offset                  //
       if (!p2.new) out_height = add(out_height,#-1)      //
     } {
       p0 = cmp.gt(out_height,#0)                         //
       s8s8s8s8.w = vadd(s8s8s8s8h.w, s8s8s8s8l.w)        //32sums of 8 final block into tmp  
     } {
       aout.w = vsub(zero.w, aout.w)                      //make sum -ve
       if(p1) vmem(out_ptr++#1) = aout.new                //write after 10 then 8 finished ints
       acc31_00 = vror(acc31_00, c16)                     //rotate out fo way 16bytes 
       if (p0) jump .L_post                               //
     } 
/* --------------------------------------------------------------------------------------- */
     { jumpr r31                                          //return
     }
.L_end:
/*=============================================================================*/
      .size gsum_asm, .L_end-gsum_asm
/*=============================================================================*/
#endif

