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
     Perform 2d integral over activation input of size in_depth. The in_depth blocks are collapsed to 1
     int and then the ints are accumulated horizontally and vertically.
 */
#if 0
void gvint_cn(uint8_t * in_data, int * out_sum, int in_width, int in_height, int in_depth, int *tmp_buf)
              int filt_w, int filt_h,  int stride_w, int stride_h, in out_width, out_height)
{
  int i,j,k, l;
   int sum, sumo;

      for(j=0; j < in_height; j++)
      {
          for(i=0; i < in_width; i++)
          {
             sum = 0;
             for(l = 0; l < in_depth/32; l++) {
                for(k=0; k < 32; k++) sum += in_data[in_depth*in_width*j+32*in_width*l+32*i+k];
             }
             tmp_buf[i] = sum;
          }

          sum = 0;
          for(i=0; i < in_width;  i++)
          {
             sum += tmp_buf[i];
             if(j==0) sumo = sum; else sumo = sum + out_sum[(j-1)*in_width + i];
             out_sum[in_width*j+i] = sumo;
          }
      }
}
      integrate in front of the  z = in_depth * filt_offset
       z 2z 3z 4z in front of every integral
       
#endif
/* ------------------------------------------------------------------------------------------ */
        .text
        .file "gvint_h.S"		
        .global gvint_asm
        .balign 32
        .type  gvint_asm, @function
gvint_asm:
/* ------------------------------------------------------------------------------------------ */
#define in_ptr0                     r0     //
#define out_ptr0                    r1     //
#define next_d32_row                r2     //width of image physical
#define next_input                  r3     //mpyi(in_depth, in_width_bytes) logical width of image
#define integral_width              r4     //number of integral outputs
#define in_depth                    r5     //indepth multiple of 32  
#define out_height                  r22    //number of required output rows
#define tmp_buf0                    r23    //tmp buffer
#define filt_offset                 r20    //filteroffset * data sum
#define cntrl                       r11    //cntrl = ##integral_cntrl  
/* ------------------------------------------------------------------------------------------ */
#define in_count                    r25
#define next_int_width              r6     //distance to next output of integral buffer
#define c4                          r7     //
#define e1                          r17
#define e2                          r18
#define e3                          r8
#define e4                          r9
#define e5                          r10
#define in_ptr                      r12 
#define out_ptr                     r13 
#define out_ptr_1                   r14
#define n                           r15      //loop count
#define tmp_buf                     r19
#define next_output                 r21      //jmp required to advance to next lot of computation
/* ------------------------------------------------------------------------------------------ */
#define vq1                         q0
#define vq2                         q1
#define vq3                         q2
#define vq4                         q3
#define vq5                         q3
#define preds                       v27
#define perm1                       v0
#define perm2                       v1
#define perm3                       v2
#define perm4                       v3
#define perm5                       v4
#define perm6                       v5
#define delta4                      v6
#define delta8                      v6
#define delta16                     v6
#define delta32                     v6
#define delta64                     v6
#define delta128                    v7
#define vzero                       v26
#define d3d2d1d0                    v9
#define h16g16f16e16h_h16g16f16e16l v11:10
#define h16g16f16e16h               v11
#define h16g16f16e16l               v10
#define d16c16b16a16h_d16c16b16a16l v31:30
#define d16c16b16a16h               v31
#define d16c16b16a16l               v30
#define h32g32f32e32h_h32g32f32e32l v13:12
#define h32g32f32e32h               v13
#define h32g32f32e32l               v12
#define a8a8a8a8                    v22
#define b8b8b8b8                    v23
#define b8b8a8a8h_b8b8a8a8l         v29:28
#define b8b8a8a8h                   v29
#define b8b8a8a8l                   v28
#define b16b16a16a16                v25
#define d16d16c16c16                v22
#define f16f16e16e16                v14
#define h16h16g16g16                v16
#define d32c32b32a32                v18
#define h32g32f32e32                v19
#define intw31w00                   v20
#define inty31y00                   v24
#define prev_line                   v21
/* --------------------------------------------------------------------------------------- */
     { allocframe(#4*8)                                                   //
       c4 = #-4                                                           //
       cntrl = add(pc,##integral_control@PCREL)                           //
     } {
       memd(sp+#0) = r17:16                                               //
       memd(sp+#8) = r19:18                                               //
       vzero = #0                                                         //
       n = lsr(integral_width, #5)                                        //integral_width / 32 
     } {
       memd(sp+#16) = r21:20                                              //
       memd(sp+#24) = r23:22                                              //
       n = add(n, #-1)                                                    //
     } {
       filt_offset = memw(sp+#12<<2)                                      //filt_offset unsigned 
       M0 = next_d32_row                                                  //
       next_int_width = asl(integral_width, #2)                           //line to line in integral
     } {
       tmp_buf0 = memw(sp+#11<<2)                                         //splat filt offset 
       preds = vmem(cntrl+#6)                                             //
       filt_offset = vsplatb(filt_offset)                                 //
       in_ptr  = in_ptr0                                                  //
     } {
       tmp_buf = tmp_buf0                                                 //
       perm1 = vmem(cntrl+#0)                                             //
       e1 = ##0x01010101                                                  //
     } {
       vq1 = vand(preds, e1)                                              //
       perm2 = vmem(cntrl+#1)                                             //
       e2 = add(e1, e1)                                                   //
       out_height = memw(sp+#10<<2)                                       //
     } {
       vq2 = vand(preds, e2)                                              //
       perm3 = vmem(cntrl+#2)                                             //
       e3 = add(e2, e2)                                                   //
       p2 = and(p2, !p2)                                                  //disable p2 for previous line
     } {
       vq3 = vand(preds, e3)                                              //
       perm4 = vmem(cntrl+#3)                                             //
       in_depth = lsr(in_depth, #5)                                       //how many 32byte chunks to sum
     } {
       e4 = add(e3, e3)                                                   //
       perm5 = vmem(cntrl+#4)                                             //
       next_output = add(next_input, #-256)                               //jump 256 bytes right doinf 2 at once
       integral_width = lsr(integral_width, #3)                           //64 * 32/ 256 amount of tmp values
     } {
       integral_width = add(integral_width, #-1)                          //after pad to left side
       e5 = add(e4, e4)                                                   //select vpred 5
       perm6 = vmem(cntrl+#5)                                             //last word across whole re
       tmp_buf = tmp_buf0                                                 //
     }
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_height:
     //sum up and pad integral prepend 8 lines to the integral buffer and 8 elements in front
     { vmem(tmp_buf++#1) = vzero                                          //store 32 sums of 8*inz*filtz*in_depth/32 pre-pad
       in_ptr  = in_ptr0                                                  //
       in_ptr0 = add(in_ptr0, next_input)                                 //[P, 3]update to next indepth*in_width 
       loop1(.L_width, integral_width)                                    //setup tmp horz loop
     }
.L_width:
     { a8a8a8a8 = #0                                                      //init accumulators
       b8b8b8b8 = #0                                                      //init accumulators
       loop0(.L_sum, in_depth)                                            //set up inner loop of horz sum
       nop                                                                //
     }
.L_sum:
     { d3d2d1d0.tmp = vmem(in_ptr+#1)                                     //col even
       b8b8b8b8.uw += vrmpy(d3d2d1d0.ub, filt_offset.ub)                  //32 sums of 4
     } {
       d3d2d1d0.tmp = vmem(in_ptr++M0)                                    //col odd
       a8a8a8a8.uw += vrmpy(d3d2d1d0.ub, filt_offset.ub)                  //32 sums of 4
     }:endloop0
     {
       loop0(.L_sum, in_depth)                                            //reset inner loop
       b8b8a8a8h_b8b8a8a8l = vdeal(b8b8b8b8, a8a8a8a8, c4)                //expand words to add in a tree
       a8a8a8a8 = #0                                                      //reset accumulator
       b8b8b8b8 = #0                                                      //reset accumulator
     } {
       in_ptr  = sub(in_ptr, next_output)                                 //next set of outputs
       b8b8a8a8h.w = vadd(b8b8a8a8h.w, b8b8a8a8l.w)                       //32sums of 8 final block into tmp  
       vmem(tmp_buf++#1) = b8b8a8a8h.new                                  //final tmp store
     }:endloop1

/* --------------------------------------------------------------------------------------- */
.L_integrate:
     { b16b16a16a16= vmem(tmp_buf0+#0)                                    //[P, 0]load 32sums of 8
     } {
       tmp_buf = add(tmp_buf0, #256)                                      //[P, 0]update ptr by 2 vregs
     } {
       d16d16c16c16.tmp= vmem(tmp_buf+#-1)                                //[0, 1]load 32sums of 8
       d16c16b16a16h_d16c16b16a16l= vdeal(d16d16c16c16, b16b16a16a16, c4) //[0, 1]deal out for 32sums of 16
     } {
       f16f16e16e16= vmem(tmp_buf++#2)                                    //[0, 2]load 32sums of 8
     } {
       out_ptr  = out_ptr0                                                //[P, 3]setup out pointer
       delta128 = #0                                                      //[P, 3]initialize add sum to next vreg
     } {
       h16h16g16g16.tmp= vmem(tmp_buf+#-1)                                //[0, 4]
       h16g16f16e16h_h16g16f16e16l= vdeal(h16h16g16g16,f16f16e16e16, c4)  //[0, 4]
     } {
       h32g32f32e32.w = vadd(h16g16f16e16h.w,h16g16f16e16l.w)             //[0, 5]32 x 16wordsums
       d32c32b32a32.w = vadd(d16c16b16a16h.w,d16c16b16a16l.w)             //[0, 5]32sums of 16
     } {
       out_ptr_1 = sub(out_ptr, next_int_width)                           //previous line
       b16b16a16a16= vmem(tmp_buf++#2)                                    //[0, 6]
     } {
       h32g32f32e32h_h32g32f32e32l= vdeal(h32g32f32e32,d32c32b32a32, c4)  //[0, 7]
     } {
       intw31w00.w = vadd(h32g32f32e32h.w, h32g32f32e32l.w)               //[0, 8]32 x 32wordsums  
       loop0(.L_loop_int, n)                                              //[P, 8]
     } 
/* --------------------------------------------------------------------------------------- */
     { d16d16c16c16.tmp= vmem(tmp_buf+#-1)                                //[1, 0]
       d16c16b16a16h_d16c16b16a16l= vdeal(d16d16c16c16, b16b16a16a16, c4) //[1, 0]
     } {
       delta4 = vdelta(intw31w00, perm1)                                  //[0,10]add words  
       f16f16e16e16= vmem(tmp_buf++#2)                                    //[1, 1]
     } {
       if(vq1) intw31w00.w += delta4.w                                    //[0,11]groups of 1word 1_1_
     } {
       h16h16g16g16.tmp= vmem(tmp_buf+#-1)                                //[1, 3]
       h16g16f16e16h_h16g16f16e16l= vdeal(h16h16g16g16,f16f16e16e16, c4)  //[1, 3]
     } {
       delta8 = vdelta(intw31w00, perm2)                                  //[0,13]2words
       h32g32f32e32.w = vadd(h16g16f16e16h.w,h16g16f16e16l.w)             //[1, 5]32 x 16wordsums
       d32c32b32a32.w = vadd(d16c16b16a16h.w,d16c16b16a16l.w)             //[1, 5]32sums of 16
     } {
       if(vq2) intw31w00.w += delta8.w                                    //[0,14]groups of 2words 11__11__
       b16b16a16a16= vmem(tmp_buf++#2)                                    //[1, 4]
     } {
       inty31y00 = intw31w00                                              //[0,16]
       h32g32f32e32h_h32g32f32e32l= vdeal(h32g32f32e32,d32c32b32a32, c4)  //[1, 7]
     } {
       delta16 = vdelta(intw31w00, perm3)                                 //[0,17]
       intw31w00.w = vadd(h32g32f32e32h.w, h32g32f32e32l.w)               //[1, 8]32 x 32wordsums  
     } 
/* --------------------------------------------------------------------------------------- */
     .balign   32
.L_loop_int:
     { if(vq3) inty31y00.w += delta16.w                                   //[0,18]/groups of 4words 1111____1111____
       d16d16c16c16.tmp= vmem(tmp_buf+#-1)                                //[2, 0]
       d16c16b16a16h_d16c16b16a16l= vdeal(d16d16c16c16, b16b16a16a16, c4) //[2, 0]
     } {
       vq4 = vand(preds, e4)                                              //[0,19]
       delta4 = vdelta(intw31w00, perm1)                                  //[1,10]add words  
       f16f16e16e16= vmem(tmp_buf++#2)                                    //[2, 1]
     } {
       delta32 = vdelta(inty31y00, perm4)                                 //[0,20]
       prev_line = vmem(out_ptr_1++#1)                                    //[0,20]
       if(vq1) intw31w00.w += delta4.w                                    //[1,11]groups of 1word 1_1_
     } {
       if(vq4) inty31y00.w += delta32.w                                   //[0,21]groups of 8words 11111111________
       h16h16g16g16.tmp= vmem(tmp_buf+#-1)                                //[2, 3]
       h16g16f16e16h_h16g16f16e16l= vdeal(h16h16g16g16,f16f16e16e16, c4)  //[2, 3]
     } {
       vq5 = vand(preds, e5)                                              //[0,22]
       delta8 = vdelta(intw31w00, perm2)                                  //[1,13]2words
       b16b16a16a16= vmem(tmp_buf++#2)                                    //[2, 4]
     } {
       delta64 = vdelta(inty31y00, perm5)                                 //[0,23]
       if(vq2) intw31w00.w += delta8.w                                    //[1,14]groups of 2words 11__11__
       h32g32f32e32.w = vadd(h16g16f16e16h.w,h16g16f16e16l.w)             //[2, 5]32 x 16wordsums
       d32c32b32a32.w = vadd(d16c16b16a16h.w,d16c16b16a16l.w)             //[2, 5]32sums of 16
     } {
       delta128 = vdelta(delta128, perm6)                                 //[0,24]full replication of last word
       if(vq5) inty31y00.w += delta64.w                                   //[0,24]groups of16words 111--111___--___
       if(!p2) prev_line = vzero                                          //[0,24]
     } {
       delta128.w = vadd(inty31y00.w, delta128.w)                         //[0,25]add previous last value 
       inty31y00 = intw31w00                                              //[1,16]
       h32g32f32e32h_h32g32f32e32l= vdeal(h32g32f32e32,d32c32b32a32, c4)  //[2, 7]
     } {
       prev_line.w = vadd(prev_line.w, delta128.w)                        //[0,26]
       vmem(out_ptr++#1) = prev_line.new                                  //[0,26]
       delta16 = vdelta(intw31w00, perm3)                                 //[1,17]
       intw31w00.w = vadd(h32g32f32e32h.w, h32g32f32e32l.w)               //[2, 8]32 x 32wordsums  
     }:endloop0 
/* --------------------------------------------------------------------------------------- */
     { if(vq3) inty31y00.w += delta16.w                                   //[2,18]/groups of 4words 1111____1111____
     } {
       vq4 = vand(preds, e4)                                              //[2,19]
       out_ptr0 = add(out_ptr0, next_int_width)                           //[E,19]go to next output line
     } {
       delta32 = vdelta(inty31y00, perm4)                                 //[2,20]
       prev_line = vmem(out_ptr_1++#1)                                    //[2,20]
     } {
       if(vq4) inty31y00.w += delta32.w                                   //[2,21]groups of 8words 11111111________
       out_height = add(out_height, #-1)                                  //
     } {
       vq5 = vand(preds, e5)                                              //[2,22]
       p0 = cmp.eq(out_height, #0)                                        //
     } {
       delta64 = vdelta(inty31y00, perm5)                                 //[2,23]
       tmp_buf = tmp_buf0                                                 //
     } {
       delta128 = vdelta(delta128, perm6)                                 //[2,24]full replication of last word
       if(vq5) inty31y00.w += delta64.w                                   //[2,24]groups of16words 111--111___--___
       if(!p2) prev_line = vzero                                          //[1,24]
     } {
       delta128.w = vadd(inty31y00.w, delta128.w)                         //[2,25]add previous last value 
       p2 = cmp.eq(r0,r0)                                                 //enable p2
     } {
       prev_line.w = vadd(prev_line.w, delta128.w)                        //[2,26]
       vmem(out_ptr++#1) = prev_line.new                                  //[2,26]
       if(!p0) jump:t .L_height                                           //[E,26]
     }
/* --------------------------------------------------------------------------------------- */
     { r17:16 = memd(sp+#0) 
       r19:18 = memd(sp+#8) 
     } {
       r21:20 = memd(sp+#16)
       r23:22 = memd(sp+#24)
     } {
       dealloc_return
     }
/* --------------------------------------------------------------------------------------- */
.L_end:		
      .size gvint_asm, .L_end-gvint_asm
		
