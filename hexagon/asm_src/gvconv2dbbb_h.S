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

/*======================================================================*/
/*  FUNCTIONS      : gvmmpybbw_asm                                      */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Perform gvm vector matrix multiply, result left at   */
/*                 32bits                                               */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*  DJH                 05/10/16       added post add for x and y offset*/
/*  DJH                 07/10/16       rewrote pre-transpose            */
/*  DJH                 09/16/16       fix over prefetch by 16 now 8    */
/*======================================================================*/
/*  CYCLE-COUNT:                                                        */
/*     ->  K*N/256+11*N/4+24                                            */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 960 bytes                                             */
/*     STACK    = 48 bytes                                              */
/*     ASSUMPTIONS                                                      */
/*        y and z are 128 byte aligned                                  */
/*        x is 8byte aligned                                            */
/*        N%4=0 K%16=0 M%32=0                                           */
/*  C MODEL                                                             */
/*======================================================================*/
#if 0
void gvmmpybbw_cn(uint8 * a, uint8 * b, int * c, int N, int M, int K) {
    int i, j, k;
    int32 sum;
    uint8 a_val, b_val;

    for (j=0; j < M; j++) {
        for (i=0; i < N; i++) {
            sum = 0;
            for (k=0; k < K; k++) {
              a_val = a[i*K+k];
              b_val = b[k*M+j];
              sum  += a_val * b_val ;
            }
            c[i*M+j] = sum;
        }
    }
    return;
}
#endif
/*=============================================================================*/
        .text
        .file "gvconv2dbbb_h.S"
        .global gvconv2dbbb_asm
        .balign 32
        .type  gvconv2dbbb_asm, @function
gvconv2dbbb_asm:
/*=============================================================================*/
#define ptr_x         r0    //data
#define ptr_yi        r1    //weights
#define ptr_z         r2    //results
#define in_width      r3    //(pad_x+in_width) * depth
#define out_width     r4    //out_width 
#define m             r5    //is stride of the output matrix always mult of 32
#define stride_depth  r6    //0 stride|depth  between computations
#define filt_width    r7    //1 depth*filt_width  
#define filt_height   r8    //2 filt_hieght lines per filter
#define out_height    r9    //3 number of vertical lines to perform
#define ptr_datasum   r10   //4 
#define ptr_weightsum r11   //5 
#define ptr_max       r12   //6 
#define ptr_biasbuf   r14   //7  sat8 ((0x8000 + (x + biass)*recip_level)>>16)
#define recip_level   r15   //8 
#define PREFETCH      r28  //64
/*=============================================================================*/
#define sel           r8
#define len           r9
#define filt_skip     r13   //the skip back after the fot_width is done for next filt_y
#define stride3_1     r1
#define ptr_x0        r11
#define stride4       r13   //
#define stride        r25
#define next_outputs  r23   //jump to input ptr for next set of outputs
#define ptr_y         r24   //
#define col_count     r22
#define xsum          r0     //kernel sum * filt_offset computed externally
#define pre_x         r26
#define fetch_count   p2 //r27 is free !
#define c4            r6 
#define round_amt     r6     //amount to add to bias buf odffset computation
#define sel0          r16
#define sel1          r18
#define sel2          r19
#define sel3          r20
#define one           r17
#define tmp_ptr_z     r21

#define MSTRIDE       M0     //stride*depth
#define M4STRIDE_1    M1     //3*stride*depth-16     0-1-2-3

                             //01234567
#define x07x04x03x00  r21:20 //11-----1
#define x07x04        r21    //11-----1
#define x03x00        r20    //1------1
#define x0fx0cx0bx08  r15:14 //1111---1
#define x0fx0c        r15    //1111---1
#define x0bx08        r14    //111----1
#define x17x14x13x10  r19:18 //11------
#define x17x14        r19    //11------
#define x13x10        r18    //1-------
#define x1fx1cx1bx18  r17:16 //1111----
#define x1fx1c        r17    //1111----
#define x1bx18        r16    //111-----
#define x27x24x23x20  r21:20 //---111--
#define x27x24        r21    //---111--
#define x23x20        r20    //---11---
#define x2fx2cx2bx28  r19:18 //---1111-
#define x2fx2c        r19    //---11111
#define x2bx28        r18    //---1111-
#define x37x34x33x30  r15:14 //----11--
#define x37x34        r15    //----11--
#define x33x30        r14    //----1---
#define x3fx3cx3bx38  r17:16 //----1111
#define x3fx3c        r17    //----1111
#define x3bx38        r16    //----111-
/*=============================================================================*/
#define z0            v0     //
#define z1            v1     //
#define z1z0          v1:0   //
#define z2            v2     //
#define z3            v3     //
#define z3z2          v3:2   //
#define x0            v4     //
#define x1            v5     //
#define x2            v6     //
#define x3            v7     //
#define y0            v8     //
#define y1            v9     //
#define y2            v10    //
#define y3            v11    //
#define vwsum         v15    //
#define maxomaxe      v13:12 //
#define maxe          v12    //
#define maxo          v13    //
#define vc8000        v14    //
#define biasvec       v18    //
#define recipvec      v16    //
#define rndvec        v17    //
#define vpreds        v19    //
/*=============================================================================*/
       {
           sel = ##0x01010101                     // entry 0
           len = #32                              //
           dcfetch(ptr_x)
       } {
           q0 = vsetq(len);                       // 1000
           len = #64                              //
           round_amt = ##0x00008000               //
       } {
           vpreds = vand(q0, sel)                 //
           q2 = vsetq(len);                       // 1100
           len = #96                              //
           rndvec = vsplat(round_amt)             //
       } {
           q1 = and(q2, !q0)                      // 0100
           q3 = vsetq(len)                        // 1110
           sel = add(sel, sel)                    //02020202
           dcfetch(ptr_x+#32)
       } {
           vpreds|= vand(q1, sel)                 //
           q2 = and(q3, !q2)                      // 0010
           q3 = not(q3)                           // 0001
           sel = add(sel, sel)                    //04040404
       } {
           vpreds|= vand(q2, sel)                 //
           sel = add(sel, sel)                    //08080808
           dcfetch(ptr_x+#64)
       } {
           vpreds|= vand(q3, sel)                 // entry 3 10101010 selects all zero
           stride_depth = memw(sp+#0<<2)          //extract stride*depth 
           filt_width = memw(sp+#1<<2)            //extract filt_width*depth 
       } {   
           filt_height = memw(sp+#2<<2)           //extract filt_height 
           out_height = memw(sp+#3<<2)            //number of output lines
           p0 = cmp.eq(filt_width, #1)
       } {
           ptr_datasum = memw(sp+#4<<2)           //data sum ptr
           ptr_weightsum = memw(sp+#5<<2)         //ptr pre computed weight sum
           filt_width = mpy(filt_width.L, stride_depth.L)
       } {
           ptr_max = memw(sp+#6<<2)               //ptr pre computed max value in output
           ptr_biasbuf = memw(sp+#7<<2)           //read in the ptr to the bias buffer value 
       } {
           biasvec = vmem(ptr_biasbuf+#0)         //
           recip_level = memw(sp+#8<<2)           // 
           p3 = cmp.gt(filt_width, #192)
       } {
           recipvec = vsplat(recip_level)         //
           allocframe(#72)                        //
           PREFETCH =  #96
       } {
           memw(sp+#68) = r28
           if(p0) PREFETCH =  add(PREFETCH, #-32)
       } {
           memd(sp+#32) = r25:24                  //
           memd(sp+#0)  = r17:16                  //
           stride = lsr(stride_depth, #16)        //
       } {
           memd(sp+#16) = r21:20                  //
           memd(sp+#24) = r23:22                  //
           stride_depth = mpy(stride_depth.H, stride_depth.L)
       } {
           M0 = stride_depth                      //
           memd(sp+#8)  = r19:18                  //
           memd(sp+#40) = r27:26                  //
       } {
           memw(sp+#48) = ptr_x                   //
           memw(sp+#52) = ptr_yi                  //
           r17 = asl(stride_depth, #2)
       } {
           if(!p3) PREFETCH = r17                 //skip full block
       } {
           vwsum = vmem(ptr_weightsum+#0)         //
           stride3_1 = addasl(stride_depth, stride_depth,#1)  //3*stride
           r16 = ##0x80000001                     //max negative
       } {
           stride3_1 = sub(#16, stride3_1)        //
           next_outputs = mpyi(filt_height, in_width) 
           vc8000 = vsplat(r16)                   //
           memw(sp+#56) = out_width               //
       } {
           M1 = stride3_1                         // add to
           stride4= asl(stride_depth, #1)         //4-2*stride to corrct for outper pipeline
           stride3_1 = add(stride3_1, #16)        //used for dc prefetch
           //p3 = cmp.gt(stride_depth, #96)         //is !(D <= 96) heuristic to fix prefetch
       } {
           memw(sp+#60) = m                       //
           next_outputs = sub(next_outputs, stride4)
           filt_skip = sub(filt_width, in_width)
           filt_width = lsr(filt_width, #4)       //filt_width / 16
       } {
           maxe= vmem(ptr_max+#0)
           in_width = mpyi(in_width, stride)      //
           filt_width = add(filt_width, #-1)
           stride3_1 = sub(stride3_1, stride_depth) //32 - 4*stride_depth used for dc prefetch
       }
/*============================================================================*/
        .balign 32
.L_height:
       { 
           ptr_x0 = memw(sp+#48)                  //ptr_x 
           out_height = add(out_height, #-1)      //
       } {
           col_count = memw(sp+#56)               //out_width 
           memw(sp+#48) += in_width               //ptr_x=add(ptr_x,in_width) //ptr_x+=in_width
           pre_x = add(ptr_x0, PREFETCH)
       }
        .balign 32
.L_width:
       {
           ptr_y = memw(sp+#52)                   //ptr_y = ptr_yi initialize filter pointer
           fetch_count = !cmp.eq(r2, r2)          //#0
       } {
           loop1(.L_filt_height, filt_height)     //[P, 0]for(filt_y=0; filt_y < n; filt_y+=1){
           y0 = vmem(ptr_y++#2)                   //[0, 0]32x4
           dcfetch(pre_x) 
       } {
           loop0(.L_filt_width, filt_width)       //[P, 0]ki is k1/16 - 1
           y1 = vmem(ptr_y+#-1)                   //[0, 1]32x4
           pre_x = add(pre_x, stride_depth)
       } {
           z1z0 = vcombine(vwsum, vwsum)          //[P, 0]
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[0, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[0, 2]
       } {
           z3z2 = vcombine(vwsum, vwsum)          //[P, 0]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[0, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[0, 3]
       } 
        .balign 32
.L_filt_height:
       {
           z0.uw += vrmpy(y0.ub, x03x00.ub)       //[0, 4]
           z1.uw += vrmpy(y0.ub, x13x10.ub)       //[0, 4]
           y2 = vmem(ptr_y++#2)                   //[0, 4]32x4
           dcfetch(pre_x) 
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)       //[0, 5]
           z1.uw += vrmpy(y1.ub, x17x14.ub)       //[0, 5]
           y3 = vmem(ptr_y+#-1)                   //[0, 5]32x4
           pre_x = add(pre_x, stride_depth)
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)       //[0, 6]
           z1.uw += vrmpy(y2.ub, x1bx18.ub)       //[0, 6]
       } {
           fetch_count = not(fetch_count)         //[0, 6]
           if(fetch_count) pre_x = add(pre_x, stride3_1)   //[0, 6.5]
           x2fx2cx2bx28 = memd(ptr_x0+#8)         //[0, 6.5]
           x27x24x23x20 = memd(ptr_x0++MSTRIDE)   //[0, 6.5]
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)       //[0, 7]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)       //[0, 7]
       } {
           p0 = cmp.eq(filt_width,#0)
           if (p0.new) jump:nt .L_skip
           x3fx3cx3bx38 = memd(ptr_x0+#8)         //[0, 7]
           x37x34x33x30 = memd(ptr_x0++M4STRIDE_1)//[0, 7]
       } 
       .balign 32
.L_filt_width:
       {
           z2.uw += vrmpy(y0.ub, x23x20.ub)       //[0, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)       //[0, 8]
           y0 = vmem(ptr_y++#2)                   //[1, 0]32x4
           dcfetch(pre_x) 
       } {
           z2.uw += vrmpy(y1.ub, x27x24.ub)       //[0, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)       //[0, 9]
           y1 = vmem(ptr_y+#-1)                   //[1, 1]32x4
           pre_x = add(pre_x, stride_depth)
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)       //[0,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)       //[0,10]
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[1, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[1, 2]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)       //[0,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)       //[0,11]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[1, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[1, 3]
       } {
           z0.uw += vrmpy(y0.ub, x03x00.ub)       //[1, 4]
           z1.uw += vrmpy(y0.ub, x13x10.ub)       //[1, 4]
           y2 = vmem(ptr_y++#2)                   //[1, 4]32x4
           dcfetch(pre_x) 
       } {
           z0.uw += vrmpy(y1.ub, x07x04.ub)       //[1, 5]
           z1.uw += vrmpy(y1.ub, x17x14.ub)       //[1, 5]
           y3 = vmem(ptr_y+#-1)                   //[1, 5]32x4
           pre_x = add(pre_x, stride_depth)
       } {
           z0.uw += vrmpy(y2.ub, x0bx08.ub)       //[1, 6]
           x2fx2cx2bx28 = memd(ptr_x0+#8)         //[1, 6.5]
           dcfetch(pre_x)                         //extra dcfetch
       } {
           if(fetch_count) pre_x = add(pre_x, stride3_1) //[0, 6.5]
           fetch_count = not(fetch_count)
           z1.uw += vrmpy(y2.ub, x1bx18.ub)       //[1, 6]
           x27x24x23x20 = memd(ptr_x0++MSTRIDE)   //[1, 6.5]
       } {
           z0.uw += vrmpy(y3.ub, x0fx0c.ub)       //[1, 7]
           z1.uw += vrmpy(y3.ub, x1fx1c.ub)       //[1, 7]
           x3fx3cx3bx38 = memd(ptr_x0+#8)         //[1, 7]
           x37x34x33x30 = memd(ptr_x0++M4STRIDE_1)//[1, 7]
       }:endloop0 
.L_skip:
       {
           z2.uw += vrmpy(y0.ub, x23x20.ub)       //[1, 8]
           z3.uw += vrmpy(y0.ub, x33x30.ub)       //[1, 8]
           ptr_x0 = sub(ptr_x0, filt_skip)        //[E, 0]move to next line ptr_y keeps going
           y0 = vmem(ptr_y++#2)                   //[0, 0]32x4
       } {
           fetch_count = !cmp.eq(r2, r2)         //#0
           pre_x = add(ptr_x0, PREFETCH)         //
           z2.uw += vrmpy(y1.ub, x27x24.ub)       //[1, 9]
           z3.uw += vrmpy(y1.ub, x37x34.ub)       //[1, 9]
       } {
           loop0(.L_filt_width, filt_width)       //[P, 0]ki is k1/16 - 1
           y1 = vmem(ptr_y+#-1)                   //[0, 1]32x4
           dcfetch(pre_x)
           pre_x = add(pre_x, stride_depth)
       } {
           z2.uw += vrmpy(y2.ub, x2bx28.ub)       //[1,10]
           z3.uw += vrmpy(y2.ub, x3bx38.ub)       //[1,10]
           x0fx0cx0bx08 = memd(ptr_x0+#8)         //[0, 2]
           x07x04x03x00 = memd(ptr_x0++MSTRIDE)   //[0, 2]
       } {
           z2.uw += vrmpy(y3.ub, x2fx2c.ub)       //[1,11]
           z3.uw += vrmpy(y3.ub, x3fx3c.ub)       //[1,11]
           x1fx1cx1bx18 = memd(ptr_x0+#8)         //[0, 3]
           x17x14x13x10 = memd(ptr_x0++MSTRIDE)   //[0, 3]
       }:endloop1
       {
           ptr_x0 = sub(ptr_x0, next_outputs)     //#0reset data ptr to next set of 4
           xsum = memw(ptr_datasum++#1<<2)        //
           y0 = rndvec                            //out0 = 0x8000
           sel0 = extractu(ptr_z, #2, #5)         //xx00000
       } {
           x0 = vsplat(xsum)                      //
           m = memw(sp+#60)                       //
           pre_x = add(ptr_x0, PREFETCH)         //generate l1 prefetch ptr 
           one = #1                               //
       } {
           sel0 = asl(one, sel0)                  //which predicate for out 0
           z0.w = vadd(z0.w, x0.w)                //add data sum
           p0 = cmp.gt(col_count, #1)             // are there at least 2 levt?
           y1 = rndvec                            //#1out1 = 0x8000
       } {
           maxe.w = vmax(maxe.w, z0.w)            //see if z0 is max
           z0.w = vadd(z0.w, biasvec.w)           //add data sum
           sel0 = vsplatb(sel0)                   // 01 -> 01010101  
           if(p0) xsum = memw(ptr_datasum++#1<<2) // #1
       } {
           dcfetch(ptr_x0)                        //
           q0 = vand(vpreds, sel0)                //
           x1 = vsplat(xsum)                      //#1
           p1 = cmp.gt(col_count, #2)             //#2
       } {
           y0.w += vmpyie(z0.w, recipvec.uh)      //
           z1.w = vadd(z1.w, x1.w)                //#1
           if(p1) xsum = memw(ptr_datasum++#1<<2) //#2
           y2 = rndvec                            //#2out2 = 0x8000
       } {
           x1.w = vadd(z1.w, biasvec.w)           //#1add data sum
           if(!p0) z1 = vc8000                    //#1
           x2 = vsplat(xsum)                      //#2
           p2 = cmp.gt(col_count, #3)             //#3
       } {
           y0.h = vpacko(y0.w, y0.w)              //>>16  
           maxe.w = vmax(maxe.w, z1.w)            //#1
           z2.w = vadd(z2.w, x2.w)                //#2
           if(p2) xsum = memw(ptr_datasum++#1<<2) //#3
       } {
           y1.w += vmpyie(x1.w, recipvec.uh)      //#1
           x2.w = vadd(z2.w, biasvec.w)           //#2add data sum
           if(!p1) z2 = vc8000                    //#2
           dcfetch(ptr_x0+#32)                    //
       } {
           y3 = rndvec                            //#3out3 = 0x8000
           y0.ub = vpack(y0.h, y0.h):sat          //sat8  <0, >255  
           tmp_ptr_z = add(ptr_z, m)              //
           x3 = vsplat(xsum)                      //#3
       } {
           sel1 = extractu(tmp_ptr_z, #2, #5)     //#1
           one = mux(p0, #1, #16)                 //#1
           y2.w += vmpyie(x2.w, recipvec.uh)      //#2
       } {
           sel1 = asl(one, sel1)                  //#1
           y1.h = vpacko(y1.w, y1.w)              //#1>>16  
           maxe.w = vmax(maxe.w, z2.w)            //#2
           z3.w = vadd(z3.w, x3.w)                //#3
       } {
           if(q0) vmem(ptr_z+#0):nt = y0          //[E,  ]store first 32bytes
           sel1 = vsplatb(sel1)                   //#1
           y2.h = vpacko(y2.w, y2.w)              //#2>>16  
           x3.w = vadd(z3.w, biasvec.w)           //#3add data sum
       } {
           dcfetch(ptr_x0+#64)                    //
           q1 = vand(vpreds, sel1)                //#1
           y1.ub = vpack(y1.h, y1.h):sat          //#1sat8  <0, >255 
           ptr_z = add(ptr_z, m)                  //
       } {
           col_count = add(col_count, #-4)        //
           if(p0)tmp_ptr_z = add(tmp_ptr_z, m)    //#1
           y3.w += vmpyie(x3.w, recipvec.uh)      //#3
           if(!p2) z3 = vc8000                    //#3
       } {
           sel2 = extractu(tmp_ptr_z, #2, #5)     //#2
           one = mux(p1, #1, #16)                 //#2
           if(p1)tmp_ptr_z = add(tmp_ptr_z, m)    //#2
           maxe.w = vmax(maxe.w, z3.w)            //#3
       } {
           sel2 = asl(one, sel2)                  //#2
           y3.h = vpacko(y3.w, y3.w)              //#3>>16  
           sel3 = extractu(tmp_ptr_z, #2, #5)     //#3
           one = mux(p2, #1, #16)                 //#3
       } {
           if(q1) vmem(ptr_z+#0):nt = y1          //#1[E,  ]store 2nd 32bytes
           sel2 = vsplatb(sel2)                   //#2
           y2.ub = vpack(y2.h, y2.h):sat          //#2sat8  <0, >255 
           sel3 = asl(one, sel3)                  //#3    
       } {
           if(p0)ptr_z = add(ptr_z, m)            //#1
           q2 = vand(vpreds, sel2)                //#2
           y3.ub = vpack(y3.h, y3.h):sat          //#3sat8  <0, >255 
           sel3 = vsplatb(sel3)                   //#3 08 -> 08080808
       } {
           if(q2) vmem(ptr_z+#0):nt = y2          //#2[E,  ]store 2nd 32bytes
           if(p1)ptr_z = add(ptr_z, m)            //#2
           q3 = vand(vpreds, sel3)                //#3
       } {
           if(q3) vmem(ptr_z+#0):nt = y3          //#3[E,  ]store 2nd 32bytes
           if(p2)ptr_z = add(ptr_z, m)            //#3
           p3 = cmp.gt(col_count, #0)             //
           if(p3.new) jump:t .L_width             //
       }//end cols per line
       {
           p1 = cmp.eq(out_height, #0)            //
           if(!p1.new) jump:t .L_height           //
       }//end lines per block
       {
           loop0(.L_peak, #5)                     //[P, 0]
           c4 = #4                                //
       }
.L_peak:
       {
           maxomaxe=vshuff(maxe,maxe,c4)          //[0, 0]
       } {
           maxe.w = vmax(maxo.w, maxe.w)          //[0, 1]
           c4 = add(c4, c4)                       //[0, 1]
       }:endloop0
       {   vmem(ptr_max+#0) = maxe                //[E, 0]
       }
/*=============================================================================*/
       {   r17:16 = memd(sp+#0)                   //restore stack
           r19:18 = memd(sp+#8)                   //Q
       } {
           r21:20 = memd(sp+#16)                  //Q
           r23:22 = memd(sp+#24)                  //Q
       } {    
           r25:24 = memd(sp+#32)                  //Q
           r27:26 = memd(sp+#40)                  //Q
       } {
           r28 = memw(sp+#68)                     //Q
       } {
           dealloc_return                         //Q
       }
.L_end:
/*=============================================================================*/
      .size gvconv2dbbb_asm, .L_end-gvconv2dbbb_asm
