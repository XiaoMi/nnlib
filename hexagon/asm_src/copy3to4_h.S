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
#if 0
void copy3to4(uint8_t * out4, uint8_t * in3, int n, int in_offset)
{
   int i;
   for(i=0; i < n; i++)
   {
       out4[4*i+0] = in3[3*i+0];
       out4[4*i+1] = in3[3*i+1];
       out4[4*i+2] = in3[3*i+2];
       out4[4*i+3] = in_offset;
   }
   return;
}
#endif
/*======================================================================*/
            .text
            .global copy3to4_asm
            .balign 32
            .type  copy3to4_asm, @function
copy3to4_asm:
/* ------------------------------------------------------------------- */
#define  out4        r0
#define  in3         r1
#define  elemns      r2
#define  filt_offset r3
#define  cntrl_tab   r4
/* ------------------------------------------------------------------- */
#define  vpred       v0
#define  vperm34     v1
#define  vin_offset  v2
#define  d95_d00     v3
#define  d127_d000   v4
/* ------------------------------------------------------------------- */
         {  r5 = ##0x01000000                     //4th byte mux in filt_offset
            loop0(.L_loop34, elemns)
            r7 = #96                              //input does 96 bytes
         } {
            M0 = r7
            vperm34 = vmem(cntrl_tab+#0) 
            vpred = vsplat(r5)                    //vmem(cntrl_tab+#1)
         } {
            filt_offset = vsplatb(filt_offset)
            d95_d00 =  vmemu(in3++M0)             //increment by 96
            r7 = ##0x01010101                     //set up vpredicate
         } {
            q0 = vand(vpred, r7)
            vin_offset = vsplat(filt_offset)      //mux in vector
         }
         .balign 32
/* ------------------------------------------------------------------- */
.L_loop34:
         {  d127_d000 = vdelta(d95_d00, vperm34)  //choose 4th byte
         } {
            d127_d000 = vmux(q0,vin_offset,d127_d000)
            d95_d00 =  vmemu(in3++M0)             //increment by 96
         } {
            vmem(out4++#1) = d127_d000            //store 128bytes
         }:endloop0
/* ------------------------------------------------------------------- */
         {  jumpr r31
         }
.L_end:
      .size copy3to4_asm, .L_end-copy3to4_asm
