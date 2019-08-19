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
        .file "fullconnlayerbatch_h.S"
        .global fullconnlayerbatch_asm 
        .balign 32
        .type  fullconnlay5rbatch_asm, @function
fullconnlayerbatch_asm:

#define  pptr_xi       r0       //pointer to array of pointers to activation batches     
#define  ptr_wi        r1       //pointer to weight chunkoutput depth 32
#define  pptr_zi       r2       //pointer to array of pointers to output
#define  in_depth_pad  r3       //the number of elements in the input vector % 32 out depth by def 32
#define  num_batches   r4       //mult of 2 if odd ptr to alias
#define  ptr_max       r5       //the on going max and mins for the output
#define  recip_level   r6       //32bit coefficients 255 / (est_max - est_min)
#define  bias_adjust   r7       //typically sum of bias and sum of weights
#define  actns_adjust  r8       //used toadjust the product
#define  woffset       r26      //the byte offset from weight position

#define  batch_cnt     r9       //number of batches to go
#define  ptr_w         r7       //current ptr to weights matrix
#define  n             r3       //num iterations
#define  cntrl         r9       //control value for populating vpredicate table
#define  sel           r27      //choose entry to predicate tab;e
#define  xsum0         r6       //sum of activations batch 0
#define  xsum1         r4       //sum of activations batch 1
#define  ptr_x0        r10      //ptr to activations 0
#define  ptr_x1        r11      //ptr to activations 1
#define  d07654_d03210 r13:12   //actviastion values 
#define  d07654        r13      //actviastion values 
#define         d03210 r12      //actviastion values 
#define  d17654_d13210 r15:14   //actviastion values 
#define  d17654        r15      //actviastion values 
#define         d13210 r14      //actviastion values 
#define  d0fedc_d0ba98 r17:16   //actviastion values 
#define  d0fedc        r17      //actviastion values 
#define         d0ba98 r16      //actviastion values 
#define  d1fedc_d1ba98 r19:18   //actviastion values 
#define  d1fedc        r19      //actviastion values 
#define         d1ba98 r18      //actviastion values 
#define  fetch0        r20      //fetch ptr to even batch
#define  fetch1        r21      //fetch ptr to odd batch
#define  out_ptr0      r22      //even output batch ptr
#define  out_ptr1      r23      //align output batch ptr
#define  align0        r24      //output alignment
#define  align1        r25      //output alignment
#define  FETCH_INC     #32      //
#define  NULL          #0       //NULL 000 ptr
#define  RSS  <<1:rnd:sat:shift //unverbose the insturction
#define PV(VSRC) .word (0x1DFFE020+VSRC)
#define PS(SSRC) .word (0x1DFFE100+SSRC) //debug reg

#define  bias0         v12      //even bias values
#define  bias1         v13      //odd bias values
#define  sum00         v1       //even accumulator
#define  sum10         v2       //odd accumulator
#define  weight00      v3       //weights from fc layer
#define  weight01      v3       //weights from fc layer
#define  weight02      v3       //weights from fc layer
#define  weight03      v3       //weights from fc layer
#define  min0          v4       //min of accumulator
#define  max0          v5       //max of accumulator
#define  max0_min0     v5:4
#define  min1          v6       //min of accumulator
#define  max1          v7       //max of accumulator
#define  max1_min1     v7:6
#define  b0            v10      //even batch quantized outputs 
#define  b1            v11      //odd batch quantized outputs
#define  vpred         v0       //sum of vector predicates for cntrling output
#define  wsum          v8       //bias values
#define  recipvec      v9       //255/max quantized value
/* ----------------------------------------------------------------------- */
             {   allocframe(#56)                           //reserve stack
                 num_batches = lsr(num_batches, #1)        //itertn 2 batches per rounf
             } {
                 memd(sp+#0) = r17:16                      //
                 memd(sp+#8) = r19:18                      //
                 loop1(.L_batches, num_batches)            //
             } {
                 memd(sp+#16) = r21:20                     //
                 memd(sp+#24) = r23:22                     //
                 n = lsr(in_depth_pad, #4)                 //
             } {
                 memd(sp+#32) = r25:24                     //
                 memd(sp+#40) = r27:26                     //
                 n = add(n, #-1)                           //correct for pipeline
             } {
                 max0 = vmem(ptr_max+#0 )                  //
		 loop0(.L_matmul32, n)                     //[P, ]]
                 cntrl = #32                               //
                 ptr_x0 = memw(pptr_xi++#1<<2)             //[P,  ] 
             } {
                 ptr_x1 = memw(pptr_xi++#1<<2)             //[P,  ]
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
                 vpred |= vand(q0, sel)                    //_1__ v(64) & !v(32)
                 sel = add(sel, sel)                       //
                 q0 = vsetq(cntrl)                         //
                 batch_cnt = add(num_batches, #-1)         //   
             } {
                 woffset = memw(sp+#19<<2)                 //
                 max1_min1 = vcombine(max0,min0)           //
                 //min1 = min0                               //
                 q1 = and(q0, !q1)                         //
                 dcfetch(ptr_x0+#0<<6)                     //[P,  ]
             } {
                 vpred |= vand(q1, sel)                    //__1_ v(96) & !v(64)
                 sel = add(sel, sel)                       //
                 q1 = not(q0)                              //
                 dcfetch(ptr_x1+#0<<6)                     //[P,  ]
             } {
                 vpred |= vand(q1, sel)                    //___1 !v(96)
                 sel = add(sel, sel)                       //
                 q1 = and(q0, !q0)                         //
                 fetch0 = add(ptr_x0, FETCH_INC)           //[
             } {
                 vpred |= vand(q1, sel)                    //sel = 0x10101010
                 fetch1 = ptr_x1                           //[P,  ]
                 xsum0 = memw(actns_adjust++#2<<2)         //[P,  ]batch 0 -sum
                 xsum1 = memw(actns_adjust+#1<<2)          //[P,  ]batch 1 - sum
              }
/* ------------------------------------------------------------------------------ */
          .balign 32
.L_batches:
              {  d07654_d03210 = memd(ptr_x0++#1<<3)       //[0, 0]read batch 0
                 d17654_d13210 = memd(ptr_x1++#1<<3)       //[0, 1]read batch 1
                 sum00 = vsplat(xsum0)                     //[P,  ]splat the sum of accs                   
                 sum10 = vsplat(xsum1)                     //[P,  ]splat the sum of accs
              } {
                 ptr_w = ptr_wi                            //[P,  ]
                 d0fedc_d0ba98 = memd(ptr_x0++#1<<3)       //[0, 2]read batch 0
                 sum00.w = vadd(sum00.w, wsum.w)           //[P,  ]set up accumulator 0
                 sum10.w = vadd(sum10.w, wsum.w)           //[P,  ]set up accumulator 1
              } 
/* ------------------------------------------------------------------------------ */
       .balign 32
.L_matmul32:   //to be urrolled but later
              {  dcfetch(fetch0+#0<<6)                     //[0, 3]prefetch batch 0
                 fetch0 = add(fetch0, FETCH_INC)           //[0, 3]increment fetch
                 fetch1 = add(fetch1, FETCH_INC)           //[0, 3]increment fetch
                 d1fedc_d1ba98 = memd(ptr_x1++#1<<3)       //[0, 3]read batch 1
              } {
                 weight00.tmp = vmem(ptr_w++#1)            //[0, 4]read weights
                 sum00.uw += vrmpy(weight00.ub, d03210.ub) //[0, 4]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight00.ub, d13210.ub) //[0, 4]do dotproduct of acts with matrix
                 dcfetch(fetch1+#0<<6)                     //[0, 4]prefetch batch 1
              } {
                 weight01.tmp = vmem(ptr_w++#1)            //[0, 5]read weights
                 sum00.uw += vrmpy(weight01.ub, d07654.ub) //[0, 5]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight01.ub, d17654.ub) //[0, 5]do dotproduct of acts with matrix
                 d07654_d03210 = memd(ptr_x0++#1<<3)       //[1, 0]get batch input
              } {
                 weight02.tmp = vmem(ptr_w++#1)            //[0, 6]read weights
                 sum00.uw += vrmpy(weight02.ub, d0ba98.ub) //[0, 6]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight02.ub, d1ba98.ub) //[0, 6]do dotproduct of acts with matrix
                 d17654_d13210 = memd(ptr_x1++#1<<3)       //[1, 1]get batch input
              } {
                 weight03.tmp = vmem(ptr_w++#1)            //[0, 7]read weights
                 sum00.uw += vrmpy(weight03.ub, d0fedc.ub) //[0, 7]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight03.ub, d1fedc.ub) //[0, 7]do dotproduct of acts with matrix
                 d0fedc_d0ba98 = memd(ptr_x0++#1<<3)       //[1, 2]get batch input
              }:endloop0
/* ------------------------------------------------------------------------------ */
              {  weight00.tmp = vmem(ptr_w++#1)            //[1, 3]read weights
                 sum00.uw += vrmpy(weight00.ub, d03210.ub) //[1, 3]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight00.ub, d13210.ub) //[1, 3]do dotproduct of acts with matrix
                 d1fedc_d1ba98 = memd(ptr_x1++#1<<3)       //[1, 3]get batch input
              } {
                 weight01.tmp = vmem(ptr_w++#1)            //[1, 4]read weights
                 sum00.uw += vrmpy(weight01.ub, d07654.ub) //[1, 4]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight01.ub, d17654.ub) //[1, 4]do dotproduct of acts with matrix
                 out_ptr1 = memw(pptr_zi+#1<<2)            //get next output batch ptr
              } {
                 weight02.tmp = vmem(ptr_w++#1)            //[1, 5]read weights
                 sum00.uw += vrmpy(weight02.ub, d0ba98.ub) //[1, 5]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight02.ub, d1ba98.ub) //[1, 5]do dotproduct of acts with matrix
                 p1 = cmp.eq(out_ptr1, NULL)               //NULL ptr?
              } {
                 weight03.tmp = vmem(ptr_w++#1)            //[1, 6]read weights
                 sum00.uw += vrmpy(weight03.ub, d0fedc.ub) //[1, 6]do dotproduct of acts with matrix
                 sum10.uw += vrmpy(weight03.ub, d1fedc.ub) //[1, 6]do dotproduct of acts with matrix
                 out_ptr0 = memw(pptr_zi++#2<<2)           //get next output batch ptr
              } {
                 out_ptr0 = add(out_ptr0, woffset)         //add the output weights offset
                 if(!p1) out_ptr1 = add(out_ptr1, woffset) //add the output weights offset if not NULL
                 p0 = cmp.eq(batch_cnt, #0)                //are all batches computed?
              } {
                 max0.w = vmax(max0.w, sum00.w)            //[E,  ]update even max
                 max1.w = vmax(max1.w, sum10.w)            //[E,  ]update odd max
                 align0 = extractu(out_ptr0, #2, #5)       //xx00000
                 if(!p0) ptr_x0 = memw(pptr_xi++#1<<2)     //[P,  ]get next evwen batch 
              } {
                 min0.w = vmin(min0.w, sum00.w)            //[E,  ]update even min
                 min1.w = vmin(min1.w, sum10.w)            //[E,  ]update odd min
                 b0.w = vmpye(sum00.w, recipvec.uh)        //[E,  ]quantize
                 if(!p0) ptr_x1 = memw(pptr_xi++#1<<2)     //[P,  ]ptr to next patch
              } {
	         loop0(.L_matmul32, n)                     //[P,  ]next batches
                 dcfetch(ptr_x0+#0<<6)                     //[P,  ]fetch next batch
                 align0 = lsl(#1, align0)                  //convert to power of 2
                 batch_cnt = add(batch_cnt, #-1)           //decrement batch count
              } {
                 b0.w+= vmpyo(sum00.w, recipvec.h):RSS     //[E,  ]quantize
                 dcfetch(ptr_x1+#0<<6)                     //[P,  ]1st fetch
                 align1 = extractu(out_ptr1, #2, #5)       //xx00000
              } {
                 b1.w = vmpye(sum10.w, recipvec.uh)        //[E,  ]quantize
                 align0 = vsplatb(align0)                  //create table lookup cntrls
                 fetch0 = add(ptr_x0, FETCH_INC)           //initialize fetcvh
                 fetch1 = ptr_x1                           //[P,  ]initialize fetcvh
              } {
                 align1 = lsl(#1, align1)                  //convert to power of 2
                 b0.h = vpack(b0.w, b0.w):sat             //[E,  ]#>>16
                 b1.w+= vmpyo(sum10.w, recipvec.h):RSS     //[E,  ]quantize
              } {
                 b0.ub = vpack(b0.h, b0.h):sat             //[E,  ]16 to 8 sat
                 q0 = vand(vpred, align0)                  //access even alignment cntrl
                 align1 = vsplatb(align1)                  //create table lookup cntrls
              } {
                 if(q0) vmem(out_ptr0) = b0                //[E,  ]store and increment next batch
                 b1.h = vpack(b1.w, b1.w):sat             //[E,  ]pack to hwords
                 if(p1) align1 = #0 //sel                  //dont store
                 if(p1) max1_min1 = vcombine(max0,min0)    //if null ptr filter out max/min
              } {
                 q1 = vand(vpred, align1)                  //get odd alignment vpred
                 if(p1) out_ptr1 = out_ptr0                // use valid address for empty vstore
                 b1.ub = vpack(b1.h, b1.h):sat             //[E,  ]16 to 8 sat
                 if(!p0) xsum0 = memw(actns_adjust++#1<<2) //[P,  ]batch 0 -sum
              } {
                 min0.w = vmin(min0.w, min1.w)             //[E,  ]merge mins
                 max0.w = vmax(max0.w, max1.w)             //[E,  ]merge maxs
                 if(q1) vmem(out_ptr1) = b1                //[E,  ]store and increment next batch
                 if(!p0) xsum1 = memw(actns_adjust++#1<<2) //[P,  ]batch 1 - sum
              }:endloop1
/* ------------------------------------------------------------------------------ */
              {  r17:16 = memd(sp+#0)                      //restore r16, r17from stack
                 r19:18 = memd(sp+#8)                      //restore r18,r19
              } {
                 r21:20 = memd(sp+#16)                     //restore r20,r11
                 vmem(ptr_max+#0) = max0                   //[E, 0]32max
              } {
                 r23:22 = memd(sp+#24)                     //restore r22,r13
                 vmem(ptr_max+#1) = min0                   //[E, 0]32min
              } {
                 r25:24 = memd(sp+#32)                     //restore r24,r15
                 r27:26 = memd(sp+#40)                     //restore r26,r17
              } {
                 dealloc_return                            //restore fram and return
              }
.L_end:
/* ------------------------------------------------------------------------------ */
      .size fullconnlayerbatch_asm, .L_end-fullconnlayerbatch_asm
/* ------------------------------------------------------------------------------ */
