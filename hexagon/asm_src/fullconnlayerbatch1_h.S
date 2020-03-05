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
 *  FUNCTIONS      : fullconnlayerbatched     
 *                                          
 *  DESCRIPTION                            
 *    perform matrix multiply on activation and output relu data and max/min
 *                                       
 *  ARCHITECTURE   : QDSP6V60+  + HVX     
 *
 *  REVISION HISTORY:                                    
 *  =================                                   
 *                                                     
 *  Author           Date      Comments       
 *  ------------------------------------------------------------- 
 *  DJH              05/22/18  created   
 *  DJH              10/21/18  stopped overreading past read arrays
 *  -------------------------------------------------------------   */
        .text
        .file "fullconnlayerbatch1_h.S"
        .global fullconnlayerbatch1_asm 
        .balign 32
        .type  fullconnlayerbatch1_asm, @function
fullconnlayerbatch1_asm:

#define  ptr_x0        r0       //pointer to activation batch     
#define  ptr_w         r1       //pointer to weight chunkoutput depth 32
#define  ptr_zi        r2       //pointer to output
#define  in_depth_pad  r3       //the number of elements in the input vector % 32 out depth by def 32
#define  num_batches   r4       //dummy input is 1
#define  ptr_max       r5       //the on going max and mins for the output
#define  recip_level   r6       //32bit coefficients 255 / (est_max - est_min)
#define  bias_adjust   r7       //typically sum of bias and sum of weights
#define  actns_adjust  r8       //used toadjust the product
#define  woffset       r10      //the byte offset from weight position
#define  recip_shift   r12      //shift for accumulator if outputs larger

#define  n             r3       //num iterations
#define  cntrl         r9       //control value for populating vpredicate table
#define  sel           r11      //choose entry to predicate table
#define  xsum0         r8       //sum of activations batch 0
#define  d07654_d03210 r13:12   //actviastion values 
#define  d07654        r13      //actviastion values 
#define         d03210 r12      //actviastion values 
#define  d0fedc_d0ba98 r15:14   //actviastion values 
#define  d0fedc        r15      //actviastion values 
#define         d0ba98 r14      //actviastion values 
#define  fetch0        r6       //fetch ptr to even batch
#define  out_ptr0      r2       //even output batch ptr
#define  align0        r4       //output alignment
#define  FETCH_INC     #32      //
#define  NULL          #0       //NULL 000 ptr
#define  RSS  <<1:rnd:sat:shift //unverbose the insturction
#define PV(VSRC) .word (0x1DFFE020+VSRC)
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug reg

#define  bias0         v12      //even bias values
#define  sum00         v1       //even accumulator
#define  weight00      v3       //weights from fc layer
#define  weight01      v3       //weights from fc layer
#define  weight02      v3       //weights from fc layer
#define  weight03      v3       //weights from fc layer
#define  min0          v4       //min of accumulator
#define  max0          v5       //max of accumulator
#define  max0_min0     v5:4
#define  b0            v10      //even batch quantized outputs 
#define  vpred         v0       //sum of vector predicates for cntrling output
#define  wsum          v8       //bias values
#define  recipvec      v9       //255/max quantized value
#define  vshamt_vec    v14      //splat quantized shift values

/* ----------------------------------------------------------------------- */
             {   allocframe(#56)                           //reserve stack
                 n = lsr(in_depth_pad, #4)                 //
             } {
                 max0 = vmem(ptr_max+#0 )                  //
                 cntrl = #32                               //
                 n = add(n, #-1)                           //correct for pipeline
             } {
                 recip_shift = memw(sp+#20<<2)             //
                 sel = ##0x01010101                        //
                 q0 = vsetq(cntrl)                         //
             } {
                 recip_level = memw(sp+#16<<2)             //get quantize coeff
                 cntrl = #64                               //
                 vpred = vand(q0, sel)                     //1___ v(32)        
                 sel = add(sel, sel)                       //
             } {
                 recipvec = vsplat(recip_level)            //replicate to vector
                 bias_adjust = memw(sp+#17<<2)             //
                 q1 = vsetq(cntrl)                         //
                 min0 = vmem(ptr_max+#1 )                  //
             } {
                 wsum = vmem(bias_adjust+#0)               //[P,  ]
                 actns_adjust = memw(sp+#18<<2)            //
                 q0 = and(q1, !q0)                         //
                 cntrl = #96                               //
             } {
                 vshamt_vec = vsplat(recip_shift)          // 
                 vpred |= vand(q0, sel)                    //_1__ v(64) & !v(32)
                 sel = add(sel, sel)                       //
                 q0 = vsetq(cntrl)                         //
             } {
                 woffset = memw(sp+#19<<2)                 //
                 q1 = and(q0, !q1)                         //
                 dcfetch(ptr_x0+#0<<6)                     //[P,  ]
             } {
                 vpred |= vand(q1, sel)                    //__1_ v(96) & !v(64)
                 sel = add(sel, sel)                       //
                 q1 = not(q0)                              //
             } {
                 vpred |= vand(q1, sel)                    //___1 !v(96)
                 sel = add(sel, sel)                       //
                 q1 = and(q0, !q0)                         //
                 fetch0 = add(ptr_x0, FETCH_INC)           //[
             } {
		 loop0(.L_matmul32, n)                     //[P, ]]
                 vpred |= vand(q1, sel)                    //sel = 0x10101010
                 xsum0 = memw(actns_adjust+#0<<2)          //[P,  ]batch 0 -sum
              } {
                 d07654_d03210 = memd(ptr_x0++#1<<3)       //[0, 0]read batch 0
                 sum00 = vsplat(xsum0)                     //[P,  ]splat the sum of accs                   
              } {
                 d0fedc_d0ba98 = memd(ptr_x0++#1<<3)       //[0, 2]read batch 0
                 sum00.w = vadd(sum00.w, wsum.w)           //[P,  ]set up accumulator 0
                 out_ptr0 = add(ptr_zi, woffset)           //add the output weights offset
              } 
/* ------------------------------------------------------------------------------ */
       .balign 32
.L_matmul32:   
              {  dcfetch(fetch0+#0<<6)                     //[0, 3]prefetch batch 0
                 fetch0 = add(fetch0, FETCH_INC)           //[0, 3]increment fetch
                 weight00.tmp = vmem(ptr_w++#1)            //[0, 4]read weights
                 sum00.uw += vrmpy(weight00.ub, d03210.ub) //[0, 4]do dotproduct of acts with matrix
              } {
                 weight01.tmp = vmem(ptr_w++#1)            //[0, 5]read weights
                 sum00.uw += vrmpy(weight01.ub, d07654.ub) //[0, 5]do dotproduct of acts with matrix
                 d07654_d03210 = memd(ptr_x0++#1<<3)       //[1, 0]get batch input
              } {
                 weight02.tmp = vmem(ptr_w++#1)            //[0, 6]read weights
                 sum00.uw += vrmpy(weight02.ub, d0ba98.ub) //[0, 6]do dotproduct of acts with matrix
              } {
                 weight03.tmp = vmem(ptr_w++#1)            //[0, 7]read weights
                 sum00.uw += vrmpy(weight03.ub, d0fedc.ub) //[0, 7]do dotproduct of acts with matrix
                 d0fedc_d0ba98 = memd(ptr_x0++#1<<3)       //[1, 2]get batch input
              }:endloop0
/* ------------------------------------------------------------------------------ */
              {  weight00.tmp = vmem(ptr_w++#1)            //[1, 3]read weights
                 sum00.uw += vrmpy(weight00.ub, d03210.ub) //[1, 3]do dotproduct of acts with matrix
              } {
                 weight01.tmp = vmem(ptr_w++#1)            //[1, 4]read weights
                 sum00.uw += vrmpy(weight01.ub, d07654.ub) //[1, 4]do dotproduct of acts with matrix
              } {
                 weight02.tmp = vmem(ptr_w++#1)            //[1, 5]read weights
                 sum00.uw += vrmpy(weight02.ub, d0ba98.ub) //[1, 5]do dotproduct of acts with matrix
              } {
                 weight03.tmp = vmem(ptr_w++#1)            //[1, 6]read weights
                 sum00.uw += vrmpy(weight03.ub, d0fedc.ub) //[1, 6]do dotproduct of acts with matrix
              } {
                 sum00.w = vasl(sum00.w, vshamt_vec.w)
                 align0 = extractu(out_ptr0, #2, #5)       //xx00000
              } {
                 b0.w = vmpye(sum00.w, recipvec.uh)        //[E,  ]quantize
                 align0 = lsl(#1, align0)                  //convert to power of 2
              } {
                 b0.w+= vmpyo(sum00.w, recipvec.h):RSS     //[E,  ]quantize
                 align0 = vsplatb(align0)                  //create table lookup cntrls
              } {
                 max0.w = vmax(max0.w, b0.w)               //[E,  ]update even max
                 min0.w = vmin(min0.w, b0.w)               //[E,  ]update even min
                 q0 = vand(vpred, align0)                  //access even alignment cntrl
              } {
                 b0.h = vpack(b0.w, b0.w):sat              //[E,  ]#>>16
              } {
                 vmem(ptr_max+#0) = max0                   //[E, 0]32max
              } {
                 vmem(ptr_max+#1) = min0                   //[E, 0]32min
              } {
                 b0.ub = vpack(b0.h, b0.h):sat             //[E,  ]16 to 8 sat
              } {
                 if(q0) vmem(out_ptr0) = b0                //[E,  ]store and increment batch
              } {
                 dealloc_return                            //restore fram and return
              }
.L_end:
/* ------------------------------------------------------------------------------ */
      .size fullconnlayerbatch1_asm, .L_end-fullconnlayerbatch1_asm
/* ------------------------------------------------------------------------------ */
