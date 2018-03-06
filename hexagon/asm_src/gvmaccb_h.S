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
/*  FUNCTIONS      : gemaccb_asm                                        */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Sum Y matrix vertically and multiply by a_offset     */
/*                                                                      */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*  DJH                 07/10/16       modified input tranpose operation*/
/*======================================================================*/
/*  IDEAL-CYCLE-COUNT:                                                  */
/*     ->  32*K/128+13                                                  */
/*                                                                      */
/*  MEMORY                                                              */
/*     CODESIZE = 352 bytes                                             */
/*     STACK    =   0 bytes                                             */
/*     ASSUMPTIONS                                                      */
/*        y 128 byte aligned                                            */
/*        x is 8byte aligned                                            */
/*        K%8=0 M%128=0                                                 */
/*  C MODEL                                                             */
/*       K = Klen | Kstride                                             */
/*       M = Mlen | Mstride                                             */
/*   write output into blocks width same as size to save memory         */
/*======================================================================*/
#if 0
void gvmaccb_cn(uint8 * b, uint8 * c, int K, int M, int8 a_offset) {
    int j, k;
    int32 sumb;
    uint8 b_val;

    for (j=0; j < M; j++) {
        sumb = 0;
        for (k=0; k < K; k++) {
            b_val = b[k*M+j];
            sumb += b_val ;
        }
        c[i*M+j] += sumb*a_offset;
    }
    return;
}
#endif
/*======================================================================*/
        .text
        .file "gvmaccb_h.S"
        .global gvmaccb_asm
        .balign 32
        .type  gvmaccb_asm, @function
gvmaccb_asm:
/*======================================================================*/
#define ptr_y         r0     //Y matrix aligned to 128bytes
#define ptr_z         r1     //integer accumulation of row of Y * xoffset
#define k             r2     //k
#define x_offset      r3     //input offset
#define dotp          r4     //
#define c16           r5     //
/*======================================================================*/
#define z0            v0     //
#define vx_offset     v1
#define y0            v2     //
#define z1            v3     //
#define z2            v4     //
/*======================================================================*/
       {
           k = lsr(k, #2)                            //inherent /4
           vx_offset = vsplat(x_offset)              //replicate words
           dotp = ##0x01010101                       // 
       } { 
           p0 = cmp.eq(x_offset, #0)                 //
           k = add(k, #-1)                           //
           if(p0.new) jump:t .L_pass                 //
       } {
           loop0(.L_loopK, k)                        //[P, 2]ki is k/8 - 1
           y0.tmp = vmem(ptr_y++#1)                  //[P, 0]
           z0.uw = vrmpy(y0.ub, dotp.ub)             //[P, 0]
           c16 = #16                                 //
       }
/*======================================================================*/
        .balign 32
.L_loopK:
       {   
           y0.tmp = vmem(ptr_y++#1)                  //[1, 0]
           z0.uw += vrmpy(y0.ub, dotp.ub)            //[1, 0]
       }:endloop0
/*=======================================================================*/
.L_pass:
       {
           z1.w = vmpyio(z0.w, vx_offset.h)          //do full 32bit 
       } {
           z2 = vmem(ptr_z+#0)                       //
       } {
           z1.w = vasl(z1.w, c16)                    //
       } {
           z1.w += vmpyie(z0.w, vx_offset.uh)        //
       } {
           z2.w = vadd(z2.w, z1.w)                   //
           vmem(ptr_z+#0) = z2.new                   //
	}{
           jumpr r31                                 //
       }
/* ===================================================================== */
.L_end:
      .size gvmaccb_asm, .L_end-gvmaccb_asm
