/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
 *  FUNCTIONS      : ivint_asm
 *
 *  DESCRIPTION
 *    Permfrm, 2d integration of a known depth 4 image. Assumption is 
 *    input width is padded to multiple of 32. Input width == output width
 *
 *  ARCHITECTURE   : QDSP6V60  + HVX
 *
 *  REVISION HISTORY:
 *  =================
 *
 *  Author              Date           Comments
 *  -------------------------------------------------------------
 *  DJH                 09/12/17       created
 *
 *  CYCLE-COUNT: Integrates 32 depth 4 elements every 7 packets 18.3 bytes / packet
 *
 *  MEMORY
 *     CODESIZE = 320 bytes
 *     STACK    = 0 bytes
 *     ASSUMPTIONS
 *     Aligned to 128 bytes, width % 128 = 0
 */

#if 0
void ivint_cn(
    uint8_t * in_data, int * out_sum, int in_width, int in_height, int filt_offset
) {
  int i,j,k;
  int sum, sumo;
  int in_depth = 4;
  int  *tmp_buf = (int)malloc(4*in_width);

  for(j=0; j < in_height; j++)
  {
      for(i=0; i < in_width; i++)
      {
          sum = 0;
          for(k=0; k < in_depth; k++)
          {
              sum += filt_offset*in_data[in_depth*in_width*j+in_depth*i+k];
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
  free(tmp_buf);
 return;
}
#endif
/* ---------------------------------------------------------------- */
            .text
            .global ivint_asm
            .balign 32
            .type  ivint_asm, @function
ivint_asm:
/* ---------------------------------------------------------------- */
//calling registers
#define in_ptr      r0    //
#define out_ptr     r1    //
#define in_width    r2    //
#define in_height   r3    //
#define filt_offset r4    //
#define cntrl       r5    //
//scaler registers
#define e1          r6    //
#define e2          r7    //
#define e3          r8    //
#define e4          r9    //
#define e5          r10   //
#define out_ptr_1   r11   //
#define width_bytes r14   //
#define ne5         r13   //
#define lcount      r12   //
#define ne5_lcount  r13:12//
//vector registers
#define vq1         q0    //
#define vq2         q1    //
#define vq3         q2    //
#define vq4         q3    //
#define vq5         q3    //
#define preds       v18   //
#define perm1       v0    //
#define perm2       v1    //
#define perm3       v2    //
#define perm4       v3    //
#define perm5       v4    //
#define perm6       v5    //
#define delta4      v6    //
#define delta8      v7    //
#define delta16     v8    //
#define delta32     v9    //
#define delta64     v10   //
#define delta128    v11   //
#define vzero       v12   //
#define intw31w00   v13   //
#define intx31x00   v14   //
#define inty31y00   v15   //
#define d31d00      v16   //
#define prev_line   v17   //
/* ---------------------------------------------------------------- */
     {
       perm1 = vmem(cntrl+#0)                            //[P, 0]stage 1 of integration add even to odd
       filt_offset = vsplatb(filt_offset)                //[P, 0]replicate filter offset
       e1 = ##0x01010101                                 //[P, 0]entry 0 of predicates
     } {
       d31d00.tmp = vmem(in_ptr++#1)                     //[P, 1]first 32 depths of data
       d31d00.uw = vrmpy(d31d00.ub, filt_offset.ub)      //[P, 1]32 sums of 4
       width_bytes = asl(in_width, #2)                   //[P, 1]convert to bytes depth  = 4
     } {
       delta128 = #0                                     //[P, 2]carry between 32 length blocks
       in_width = lsr(in_width, #5)                      //[P, 2]in_width/32
       preds = vmem(cntrl+#6)                            //[P, 2]vector of predicate cntrls
       e2 = add(e1, e1)                                  //[P, 2]entry 1 of predictes 
     } {
       in_height = mpyi(in_height, in_width)             //[P, 3]total iterations of fused loop
       vq1 = vand(preds, e1)                             //[P, 3]set up vec redicate 1
       delta4 = vdelta(d31d00, perm1)                    //[0, 0]perform 1st shuffle even, even+odd
       intw31w00 = d31d00                                //[0, 0]lifetime renewal
     } {
       p3 = sp1loop0(.L_height, in_height)               //[P, 4]set up loop firast iteration prolog
       perm2 = vmem(cntrl+#1)                            //[P, 4]2nd cntrl of permute file
       e3 = add(e2, e2)                                  //[P, 4]3rd entry ni preidcates
       if(vq1) intw31w00.w += delta4.w                   //[0, 1]groups of 1word 1_1_
     } {
       out_ptr_1 = sub(out_ptr, width_bytes)             //[P, 5]previous line
       vq2 = vand(preds, e2)                             //[P, 5]2nd predicate pairs of words
       e4 = add(e3, e3)                                  //[P, 5]4th preicate address
       d31d00 = vmem(in_ptr++#1)                         //[0, 2]2nd 32 depth of data
     } {
       perm3 = vmem(cntrl+#2)                            //[P, 6]3rd permute groups of 8 words
       vq3 = vand(preds, e3)                             //[P, 6]set up 3rd vector predicate
       e5 = add(e4, e4)                                  //[P, 6]5th predicate entry
       delta8 = vdelta(intw31w00, perm2)                 //[0, 3]spermute pairs of words
     } {
       lcount = add(in_width, #1)                        //[P, 7]set up inner width counter
       perm4 = vmem(cntrl+#3)                            //[P, 7]4th entryof permute controls
       ne5 = not(e5)                                     //[P, 7]make ne5 not e5
       if(vq2) intw31w00.w += delta8.w                   //[0, 4]groups of 2words 11__11__
     } {
       out_ptr_1 = sub(out_ptr, width_bytes)             //[P, 8]previous line
       vzero = #0                                        //[P, 8]set up a vector 0
       perm5 = vmem(cntrl+#4)                            //[P, 8]5th permute entry groups of 16 words
       d31d00.uw = vrmpy(d31d00.ub, filt_offset.ub)      //[0, 5]32 sums of 4
     } {
       perm6 = vmem(cntrl+#5)                            //[P, 9]6th entry groups of 32 words
       out_ptr_1 = add(out_ptr_1, #-128)                 //[P, 9]correct for pipeline
       intx31x00 = intw31w00                             //[0, 6]break lifetime for sp loop
       delta16 = vdelta(intw31w00, perm3)                //[0, 6]permute groupes of 8 words
     } 
/* -------------------------------------------------------------- */ 
#if 0
     {
       if(vq3) intx31x00.w += delta16.w                  //[0, 7]groups of 4words 1111____1111____
       delta4 = vdelta(d31d00, perm1)                    //[1, 0]
       intw31w00 = d31d00                                //[1, 0]
     } {
       if(vq1) intw31w00.w += delta4.w                   //[1, 1]groups of 1word 1_1_
     } {
       delta32 = vdelta(intx31x00, perm4)                //[0, 9]
       d31d00 = vmem(in_ptr++#1)                         //[1, 2]
     } {
       vq4 = vand(preds, e4)                             //[0,10]
       delta8 = vdelta(intw31w00, perm2)                 //[1, 3]
     } {
       if(vq4) intx31x00.w += delta32.w                  //[0,11]groups of 8words 11111111________
       if(vq2) intw31w00.w += delta8.w                   //[1, 4]groups of 2words 11__11__
     } {
       d31d00.uw = vrmpy(d31d00.ub, filt_offset.ub)      //[1, 5]32 sums of 4
     } {
       inty31y00 = intx31x00                             //[0,13]
       intx31x00 = intw31w00                             //[1, 6]
       delta16 = vdelta(intw31w00, perm3)                //[1, 6]
     } 
#endif
/* -------------------------------------------------------------- */ 
     .balign   32
.L_height:
     {
       prev_line = vmem(out_ptr_1++#1)                   //[0,14]
       if(vq3) intx31x00.w += delta16.w                  //[1, 7]groups of 4words 1111____1111____
       delta4 = vdelta(d31d00, perm1)                    //[2, 0]
       intw31w00 = d31d00                                //[2, 0]
     } {
       p2 = cmp.eq(e5,ne5)                               //[0,15]enable p2
       delta64 = vdelta(inty31y00, perm5)                //[0,15]
       vq5 = vand(preds, e5)                             //[0,15]
       if(vq1) intw31w00.w += delta4.w                   //[2, 1]groups of 1word 1_1_
     } {
       lcount = add(lcount, #-1)                         //[0,16]
       if(vq5) inty31y00.w += delta64.w                  //[0,16]groups of16words 111--111___--___
       delta32 = vdelta(intx31x00, perm4)                //[1, 9]
       d31d00 = vmem(in_ptr++#1)                         //[2, 2]
     } {
       if(!p2) prev_line = vzero                         //[0,17]
       delta128.w = vadd(inty31y00.w, delta128.w)        //[0,17]add previous last value 
       vq4 = vand(preds, e4)                             //[1,10]
       delta8 = vdelta(intw31w00, perm2)                 //[2, 3]
     } {
       prev_line.w = vadd(prev_line.w, delta128.w)       //[0,18]
       if(p3)vmem(out_ptr++#1) = prev_line.new           //[0,18]
       if(vq4) intx31x00.w += delta32.w                  //[1,11]groups of 8words 11111111________
       if(vq2) intw31w00.w += delta8.w                   //[2, 4]groups of 2words 11__11__
     } {
       p0 = cmp.eq(lcount, #0)                           //[0,19]
       if(p0.new) ne5_lcount = combine(e5, in_width)     //[0,19]
       delta128 = vdelta(delta128, perm6)                //[0,19]full replication of last word
       d31d00.uw = vrmpy(d31d00.ub, filt_offset.ub)      //[2, 5]32 sums of 4
     } {
       if(p0) delta128 = vzero                           //[0,20]
       inty31y00 = intx31x00                             //[1,13]
       intx31x00 = intw31w00                             //[2, 6]
       delta16 = vdelta(intw31w00, perm3)                //[2, 6]
     }:endloop0
/* -------------------------------------------------------------- */ 
#if 0
     {
       prev_line = vmem(out_ptr_1++#1)                   //[1,14]
       if(vq3) intx31x00.w += delta16.w                  //[2, 7]groups of 4words 1111____1111____
     } {
       p2 = cmp.eq(e5,ne5)                               //[1,15]enable p2
       delta64 = vdelta(inty31y00, perm5)                //[1,15]
       vq5 = vand(preds, e5)                             //[1,15]
     } {
       lcount = add(lcount, #-1)                         //[1,16]
       if(vq5) inty31y00.w += delta64.w                  //[1,16]groups of16words 111--111___--___
       delta32 = vdelta(intx31x00, perm4)                //[2, 9]
     } {
       if(!p2) prev_line = vzero                         //[1,17]
       delta128.w = vadd(inty31y00.w, delta128.w)        //[1,17]add previous last value 
       vq4 = vand(preds, e4)                             //[2,10]
     } {
       prev_line.w = vadd(prev_line.w, delta128.w)       //[1,18]
       vmem(out_ptr++#1) = prev_line.new                 //[1,18]
       if(vq4) intx31x00.w += delta32.w                  //[2,11]groups of 8words 11111111________
     } {
       p0 = cmp.eq(lcount, #0)                           //[1,19]
       if(p0.new) ne5_lcount = combine(e5, in_width)     //[1,19]
       delta128 = vdelta(delta128, perm6)                //[1,19]full replication of last word
     } {
       if(p0) delta128 = vzero                           //[1,20]
       inty31y00 = intx31x00                             //[2,13]
     }
#endif
/* -------------------------------------------------------------- */ 
     {
       prev_line = vmem(out_ptr_1++#1)                   //[2,14]load previouls line value
     } {
       p2 = cmp.eq(e5,ne5)                               //[2,15]conditionally enable p2
       delta64 = vdelta(inty31y00, perm5)                //[2,15]replicate element 31 across
       vq5 = vand(preds, e5)                             //[2,15]extract 5ths predicate
     } {
       if(vq5) inty31y00.w += delta64.w                  //[2,16]groups of16words 111--111___--___
     } {
       if(!p2) prev_line = vzero                         //[2,17]if 1st line kill previous line add
       delta128.w = vadd(inty31y00.w, delta128.w)        //[2,17]add previous last value 
     } {
       prev_line.w = vadd(prev_line.w, delta128.w)       //[2,18]add previous carry to current block
       vmem(out_ptr++#1) = prev_line.new                 //[2,18]store final 32 integrals
     } {
       jumpr r31                                         //[E, 0]return to caller
     }
/* -------------------------------------------------------------- */ 
.L_end:
/*======================================================================*/
      .size ivint_asm, .L_end-ivint_asm
