
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
 */
#if 0
static inline void biasadd_relu_requant_hvx(
	uint8_t *out,
	const int32_t *tmp_out,
	const int32_t *biasbuf,
	const uint32_t num_patches,
	const uint32_t depth,
	const uint32_t fixed_recip_level_size)
{
	int32_t sum;
	int32_t i,j;
	int32_t outval;
	/* do 4 vectors */
	/* multiply */
	/* pack odd halves */
	/* saturate and pack */
	/* deal */
	/* deal */
	/* store a vector */
	for (j = 0; j < num_patches; j++) {
		for (i = 0; i < depth; i++) {
			sum = biasbuf[i] + tmp_out[j*depth+i];
			outval = sum * fixed_recip_level_size + (1<<15);
			outval >>= 16;
			if (outval < 0) outval = 0;
			if (outval > 255) outval = 255;
			*out++ = outval;
		}
	}
}
#endif
/*==============================================================================*/
          .global biasadd_relu_requant_nonaligned_hvx
          .type   biasadd_relu_requant_nonaligned_hvx, @function
          .balign 32
biasadd_relu_requant_nonaligned_hvx: 
/* ============================================================================ */
#define outptr            r0
#define acc_buf           r1
#define bias_buf0         r2      //aligned
#define num_patches       r3      //
#define depth             r4
#define fixed_recip_level r5

#define l2cntrl_hi        r17   
#define l2cntrl_lo        r16   
#define l2cntrl           r17:16
#define l2addr            r18

#define stride            r6
#define write_cnt         r7
#define nbias             r8
#define bias_buf          r9      //aligned
#define round             r10
#define width             r11  //write width
#define dalign            r12
#define mdsto             r13
#define sel0              r14
#define sel1              r15
#define sel               r5

#define bias_data0   v0
#define bias_data1   v1
#define bias_data2   v2
#define bias_data3   v3
#define acc_data0    v4
#define acc_data1    v5
#define acc_data2    v6
#define acc_data3    v7
#define out0         v8
#define out1         v9
#define out2         v10
#define out3         v11
#define out32        v12
#define out10        v13
#define out3210      v14
#define scale        v15
#define sum0         v16
#define sum1         v17
#define sum2         v18
#define sum3         v19
#define vpredp       v20
#define vprede       v21
#define qprolog      q0
#define qepilog      q1
#define tqprolog     q2
#define tqepilog     q3
#define d0           v22
/* ============================================================================ */
             {
               allocframe(#32)
               nbias = add(depth, #127)
               sel0 = ##0x01010101
             } {
               memd(sp+#0) = r17:16
               memd(sp+#8) = r19:18
               sel1 = add(sel0, sel0)
               write_cnt = depth
             } {
               stride = and(depth, #127)
               scale = vsplat(fixed_recip_level)
               l2cntrl_hi =asl(depth, #2) 
               l2cntrl_lo = #1
             } {
               l2cntrl_lo = combine(l2cntrl_hi.L, l2cntrl_lo.L)
               p2 = cmp.eq(stride, #0)
               stride = add(stride, #-128)
               nbias = lsr(nbias, #7)
             } {
               l2addr = addasl(acc_buf, depth, #5)
               if(p2) stride = #0
               loop1(.L_loop1, num_patches)
             } {
               round = ##0x00008000
               l2fetch(l2addr, l2cntrl)                //
             }
/*==============================================================================*/
               .balign 32
.L_loop1:
            {
               bias_buf = bias_buf0                    //
               loop0(.L_loop0, nbias)                  //
               acc_data0 = vmemu(acc_buf++#1)          //[0,0]
            } {
               bias_data0.tmp = vmem(bias_buf++#1)     //[0,1]
               sum0.w = vadd(acc_data0.w, bias_data0.w)//[0,1]
               l2addr = addasl(l2addr, depth, #2)      //
            } {
               out0 = vsplat(round)                    //[0,2]
               acc_data1 = vmemu(acc_buf++#1)          //[0,2]
            }
/*==============================================================================*/
               .balign 32
.L_loop0:
            {
               dalign = and(outptr, #127)              //C
	       out0.w += vmpyie(sum0.w,scale.uh)       //[0,3]
               bias_data1.tmp = vmem(bias_buf++#1)     //[0,3]
               sum1.w = vadd(acc_data1.w, bias_data1.w)//[0,3]
            } {
               width = write_cnt                       //C
               out1 = vsplat(round)                    //[0,4]
               acc_data2 = vmemu(acc_buf++#1)          //[0,4]
            } {
               p0 = cmp.gt(write_cnt, #127)            //C
	       out1.w += vmpyie(sum1.w,scale.uh)       //[0,5]
               bias_data2.tmp = vmem(bias_buf++#1)     //[0,5]
               sum2.w = vadd(acc_data2.w, bias_data2.w)//[0,5]
            } {
               if(p0    ) width = #128                 //C
               out2 = vsplat(round)                    //[0,6]
               acc_data3 = vmemu(acc_buf++#1)          //[0,6]
            } {
               out10.h = vpacko(out1.w, out0.w)        //[0,7]
	       out2.w += vmpyie(sum2.w,scale.uh)       //[0,7]
               bias_data3.tmp = vmem(bias_buf++#1)     //[0,7]
               sum3.w = vadd(acc_data3.w, bias_data3.w)//[0,7]
            } {
               out3 = vsplat(round)                    //[0,8]
               qprolog = vsetq(outptr)                 //C
               dalign = add(dalign, width  )           //C
            } {
               mdsto = sub(#0, outptr)                 //C
               sel = sel0                              //C
               qepilog = vsetq(dalign)                 //C
	       out3.w += vmpyie(sum3.w,scale.uh)       //[0,9]
            } {
               p1 = !cmp.gt(dalign, #127)              //C is block less than 128 bytes
               if(p1.new) sel = sel1                   //C
               acc_data0 = vmemu(acc_buf++#1)          //[1,0]
            } {
               tqprolog = or(qprolog, !qepilog)        //C
               write_cnt = add(write_cnt, #-128)       //C
               out32.h = vpacko(out3.w, out2.w)        //[0,11]
            } {
               vprede = vand(qepilog, sel0)            //C
               vpredp = vand(qprolog, sel0)            //C
               bias_data0.tmp = vmem(bias_buf++#1)     //[1,1]
               sum0.w = vadd(acc_data0.w, bias_data0.w)//[1,1]
            } {
               vpredp|= vand(tqprolog, sel1)           //C
               out3210.ub = vpack(out32.h, out10.h):sat//[0,12]
            } {
               qprolog = vand(vpredp, sel)             //C
               acc_data1 = vmemu(acc_buf++#1)          //[1,2]
               out0 = vsplat(round)                    //[1,2]
            } {
               d0 = vror(out3210, mdsto)               //C
               qepilog = vand(vprede sel)              //C
            } {
               if( qepilog) vmem(outptr+#1) = d0       //C
            } {
               if(!qprolog) vmem(outptr++#1) = d0      //C
            }:endloop0
/*==============================================================================*/
            {
               acc_buf = addasl(acc_buf, stride, #2)
               outptr = add(outptr, stride)
               write_cnt = depth
            }{
               acc_buf = add(acc_buf, #-256)
               l2fetch(l2addr, l2cntrl)                //
            }:endloop1
/*==============================================================================*/
            {
                r17:16 = memd(sp+#0)
                r19:18 = memd(sp+#8)
            } {
                dealloc_return
            }
.L_end:
/*==============================================================================*/
      .size biasadd_relu_requant_nonaligned_hvx, .L_end-biasadd_relu_requant_nonaligned_hvx
