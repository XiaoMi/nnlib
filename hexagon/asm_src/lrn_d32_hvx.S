/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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

  void lrn_d32_hvx(uint8_t const * in_ptr, int in_depth, int in_offset, int radius, int32_t * tmp_buf,
               uint8_t * lrn_ptr,
               int kappa, int sigma, int beta, int recip,  int out_offset,
               int next_d32_row, int next_logical_line, 
             width, height, int depth_rng)

  Behavioral Model
  ----------------
 */
#if 0
void lrn_f(float *ptr_x, int in_depth, int in_offset, int radius, float *tmp, float *ptr_y,
           float K, float alpha, float beta, float *omin, float *omax, int width, int height)
{
  int i, j, k, h, v, w, rows = in_depth/32;
  float  sum, sigma, u, val;

  *omin = 99999; *omax =-99999;
  for(v=0; v < height; v++)
  {
    for(h=0; h < width/4; h++)
    {
      for(i=0; i < rows; i+=1)
      {
        for(j=0; j < 128; j++)
        {
           u = ptr_x[v*width*in_depth +128*h+ i*32*width+j];
           tmp[128*i+j] = u*u;
        }
      }
      for(w = 0; w < 128; w+=32)
      for(i=0; i < rows; i++) { 
         for(j=0; j < 32; j++)
         {
            sum = 0;
            for(k=-radius; k <= radius; k++)
            {
              if((k+j) < 0 && i > 0)
                 sum += tmp[(i-1)*128+k+j+w+32];
              else if((k+j) >= 32 && i < rows-1)
                 sum += tmp[(i+1)*128+k+j+w-32];
              else if((k+j) < 32 && (k+j) >= 0)
                 sum += tmp[i*128+k+j+w];
            }
            sigma = exp2( beta*log2(K + alpha*sum));
            u = ptr_x[v*width*in_depth +128*h+ i*32*width+j+w];
            val = sigma*u; 

            if(val > *omax) *omax = val;
            if(val < *omin) *omin = val;
            ptr_y[v*width*in_depth +128*h+ i*32*width+j+w] = val;
         }//end j
      }//end i/w
    }//end h
  }//end v
}
#endif
/*
     Methods+Notes
     -------------
     step_in = (imax - imin )/255.0;
     - scaled the k vaue by step size to avoid mpy
     offset = k/(alpha*step_in*step_in);
     - scaled step size by 4 and require data to be scaled by 4 for square
     in_offset = (int) (4.0*imin/step_in + 0.0);
     - offset scaled by 16 more accurate
     - all absorbed into sigma 
        gamma =log2(alpha*step_in*step_in)-4.0; <abosrd 16 with log
        sigma = -log2(imax-imin)+log2(omax-omin) - beta * gamma ;
     -beta replaced by sigma     

     from radius sum   use cl0 and shift to covert to float

     log 2**sum_exp*mant_sum = log(mant_sum) + sem_exp
     remove leading 1 from mant_sum and becomes log2(1+dmant_sum)
     dmant_sum is 15bit and poly is done using S16 arithmetic.
     sum_exp is implemented using shift.


    Width must be a multiple of 4, and depth a multiple of 32; however
     the parameter depth_rng supports depth edge padding.
      depth_rng is d_lo  in lower 16 bits, d_hi in upper, where
      	d_lo is 0..31  = 'left padding' in first depth slice
      	d_hi =  1..32  = (32-right padding); this is the number of valid
      	                 elements in the last depth slice (if depth=32, it
      	                 also includes d_lo)
      	so depth = d_lo + (actual_depth) + 32 -d_hi
      A value of depth_rng = 0 is equivalent to (32<<16) i.e. d_lo = 0, d_hi = 32;
      this is the 'no padding' case.
      Edge padding is implemented by setting up q0 and q1:

      q0 : out of first 64: 2*d_lo are 1, rest 0; the 2nd 64 is the same
      q1 : out of first 64: 2*d_hi are 1, rest 0; the 2nd 64 is the same

      for depth_rng=0, we will have q0 = {all 0}, q1 = { all 1}
      These masks are applied before squaring the inpouts; they correct the window
      contexts for the valid outputs which are near the edge, but the outputs
      which fall within the padding are still going to be indeterminate.

     Makes use of:
       extern const int16_t lut_Log2_and_Pow2[6*64]
       		- 1st 3 are luts for Log(x-1)    (2nd order x 16)
       		- 2nd 3 are luts for Pow2(-x)    (2nd order x 16)
       extern const uint8_t const_Count64[128]
       		 = { 0,1,  63, 0, 1,   .. 63 }
       	(used for depth masking)
 */
/* -------------------------------------------------------------*/
         .text
         .file "lrn_d32_hvx.S"
         .global lrn_d32_hvx
         .balign 32
         .type lrn_d32_hvx, @function
/* -------------------------------------------------------------*/
lrn_d32_hvx:
/* -------------------------------------------------------------*/
#define in_ptr00      r0    //pointer to input data
#define in_depth      r1    //
#define in_offset     r2    //pointer to input zero
#define radius        r3    //radious of lrn - window is 2*radius + 1
#define tmp_buf0      r4    //pointer to temp bug to hold sum sq -ints
#define lrn_ptr00     r5    //pointer to output of lrn 'd data
#define kappa         r14   //K in the paper
#define sigma         r15   //bunch of corrections and junk to do with compression
#define beta          r16   //as per paper
#define recip         r27   //
#define out_offset    r17   //step size correction 
#define next_in_width r10   //physical width of array 32*width
#define next_logcl_row r22  //distance to next entire row of new depths
#define in_width      r13   //how much work per line
#define in_height     r23   //number of logical wors
#define log_exp       r26   //table to log2(1+x) and exp(-x)
/* -------------------------------------------------------------*/
#define in_ptr0       r24   //pointer to input data
#define lrn_ptr0      r25   //pointer to output of lrn 'd data
#define col_count     r20
#define rows          r9    //num slices of bread
#define rowsb         r12
#define tmp_buf       r8    //
#define lrn_ptr       r21   //pointer to output of lrn 'd data
#define cm1           r7    //= -1
#define cm4           r6    //= -4
#define cm2           r2
#define eob_count     r11
#define offset        r1
#define offset1       r7
#define in_ptr_o      r14   //in ptr for output calculation
#define in_ptr_i      r15   //in ptr for input calculation
#define c0            r1    //0
/* -------------------------------------------------------------*/
#define x3x2x1x0      v0
#define x3x2_x1x0     v29:28
#define x3x2          v29
#define x1x0          v28
#define x3_x2         v5:4
#define y3y2          v21
#define y1y0          v20
#define x2            v4
#define x3            v5
#define x1_x0         v7:6
#define x1            v7
#define x0            v6
#define vzero0        v8
#define vzero1        v9
#define vzero2        v10
#define vzero3        v11
#define x00           v3
#define x10           v12
#define x20           v13
#define x30           v18
#define x01           v14
#define x11           v15
#define x21           v16
#define x31           v17
#define xe0           v14
#define xe1           v15
#define xe1_xe0       v15:14
#define xe2           v16
#define xe3           v17
#define xe3_xe2       v17:16
#define d0            v1
#define d1            v1
#define d2            v1
#define d3            v1
#define sum0          v20
#define sum1          v21
#define sum2          v22
#define sum3          v23
#define xm0           v20
#define xm1           v21
#define xm1_xm0       v21:20
#define xm2           v22
#define xm3           v23
#define xm3_xm2       v23:22
#define xm1xm0        v20
#define xm3xm2        v21
#define xm3xm2_xm1xm0 v21:20
#define dx1dx0        v22
#define dx3dx2        v23
#define vin_offset    v19
#define vout_offset   v30
#define vzero         v24
#define vkappa        v25
#define vsigma        v26
#define vbeta         v27
#define range         v1
#define tab           v2    //lut entry
#define coef1         v29
#define coef0         v28
#define coef1_coef0   v29:28
/* -------------------------------------------------------------*/
   {
       allocframe(#56)
       q1 = or(q0,!q0)				// set to all 1
   		r6 = memw(sp+#(25-16)<<2)			// depth_rng
   } {
       memd(sp+#0) = r17:16
       memd(sp+#8) = r19:18
       q0 = and(q0,!q0);				// set to all 0;
   } {
       memd(sp+#16) = r21:20
       memd(sp+#24) = r23:22
	  p1 = cmp.eq(r6,#0)
   } {
       memd(sp+#32) = r25:24
       memd(sp+#40) = r27:26
	  if( p1 ) jump:t .L_no_depth_pad
   }
//>>> set up q0,q1 for depth range, based on r6
	{
		r6 = add(r6,r6)				// d_lo * 2 ( in ls byte )
		r7 = lsr(r6,#15)			// d_hi * 2  (in ls byte )
		r16 = add(PC,##const_Count64@PCREL)		// pointer to const vec
	} {
		r6 = vsplatb(r6)
		r7 = vsplatb(r7)
		v2 = vmem(r16+#0)		// v2 = { 0, 1, ... 63, 0,1, .. 63 }
	} {
		v0 = vsplat(r6)		// all 2*d_lo
		v1 = vsplat(r7)		// all 2*d_hi
	} {
		q0 = vcmp.gt( v0.ub,v2.ub)	// q0: i < 2*d_lo
		q1 = vcmp.gt( v1.ub,v2.ub)	// q1: i < 2*d_hi
	}
// <<<<
.L_no_depth_pad:
   {
       kappa = memw(sp+#16<<2)
       sigma = memw(sp+#17<<2)
   } {
       beta = memw(sp+#18<<2)
       recip = memw(sp+#19<<2)
   } {
       out_offset = memw(sp+#20<<2)
       recip = combine(recip.L, recip.L)
   } {
       next_in_width = memw(sp+#21<<2)                 
       next_logcl_row = memw(sp+#22<<2)                 
       log_exp = add(PC,##lut_Log2_and_Pow2@PCREL)					// pointer to poly tables
   } {
       in_width = memw(sp+#23<<2)
       in_height = memw(sp+#24<<2)
       vsigma = vsplat(sigma)
   } { 
       in_offset = combine(in_offset.L, in_offset.L)
       M0 = next_in_width
       rows = lsr(in_depth, #5)                         //number of 32depths in the depth
       vzero = #0
   } {
       vin_offset = vsplat(in_offset)
       vbeta = vsplat(beta)
       out_offset = combine(out_offset.L, out_offset.L)
   } {
       vkappa = vsplat(kappa)
       cm1 = #-1
       rowsb = add(rows, #-1)
       vout_offset = vsplat(out_offset)
   }
/* -------------------------------------------------------------*/
       .balign 32
.L_height:
   {
       col_count = in_width;
       in_height = add(in_height, #-1)
       in_ptr0 = in_ptr00
       lrn_ptr0 = lrn_ptr00
   }
.L_width:
/* -------------------------------------------------------------*/
   {
       x3x2x1x0.tmp = vmem(in_ptr0+#0)       //[0, 0]
       x3x2_x1x0=vshuff(vzero,x3x2x1x0,cm1) //[0, 0]shuffle in zeros
       cm2 = #-2                              // 
       in_ptr_i = add(in_ptr0, next_in_width) //
   } {
       cm4 = #-4
       in_ptr_o = in_ptr0
       p0 = cmp.eq(rowsb, #0)				// skip loop if rows=1
   } {
       y1y0.h = vadd(x1x0.h, vin_offset.h)  //[0, 3]
       p3 = sp1loop0(.L_depth32, rowsb)      //
       tmp_buf = tmp_buf0
   } {
       col_count = add(col_count, #-4)
       lrn_ptr = lrn_ptr0
       y3y2.h = vadd(x3x2.h, vin_offset.h)  //[0, 4]
   } {
   	  y1y0 = vmux( q0, vzero, y1y0 );	// 'left side' depth masking
   	  y3y2 = vmux( q0, vzero, y3y2 );
       if(p0) jump .L_skip
   } 
/* -------------------------------------------------------------*/
       .balign 32
.L_depth32:
   {
       x1_x0.w =  vmpy(y1y0.h, y1y0.h)      //[0, 5]x1 x0 x1 x0:x1 x0 x1 x0
       x3x2x1x0.tmp = vmem(in_ptr_i++M0)    //[1, 0]
       x3x2_x1x0=vshuff(vzero,x3x2x1x0,cm1) //[1, 0]shuffle in zeros
   } {
       x3_x2.w =  vmpy(y3y2.h, y3y2.h)      //[0, 6]x3 x2 x3 x2:x3 x2 x3 x2
       if(p3) vmem(tmp_buf+#-1) = x3        //[0, 6]
   } {
       x1_x0 = vshuff(x1, x0, cm4)          //[0, 7]
       vmem(tmp_buf++#2) = x0.new           //[0, 7]
   } {
       x3_x2 = vshuff(x3, x2, cm4)          //[0, 8]
       vmem(tmp_buf++#2) = x2.new           //[0, 8]
       y1y0.h = vadd(x1x0.h, vin_offset.h)  //[1, 3]
   } {
       vmem(tmp_buf+#-3) = x1               //[0, 9]
       y3y2.h = vadd(x3x2.h, vin_offset.h)  //[1, 4]
   }:endloop0
/* -------------------------------------------------------------*/
.L_skip:
	{
   	  y1y0 = vmux(q1, y1y0, vzero);			// 'right side' depth masking
   	  y3y2 = vmux(q1, y3y2, vzero);
	}
   {
       x1_x0.w =  vmpy(y1y0.h, y1y0.h)      //[1, 5]x1 x0 x1 x0:x1 x0 x1 x0
       vzero0 = #0
   } {
       x3_x2.w =  vmpy(y3y2.h, y3y2.h)      //[1, 6]x3 x2 x3 x2:x3 x2 x3 x2
       if(p3) vmem(tmp_buf+#-1) = x3        //[1, 6]
       vzero1 = #0
   } {
       x1_x0 = vshuff(x1, x0, cm4)          //[1, 7]
       vmem(tmp_buf++#2) = x0.new           //[1, 7]
       vzero2 = #0
   } {
       x3_x2 = vshuff(x3, x2, cm4)          //[1, 8]
       vmem(tmp_buf++#2) = x2.new           //[1, 8]
       vzero3 = #0
   } {
       vmem(tmp_buf+#-3) = x1               //[1, 9]
   } {
       vmem(tmp_buf+#-1) = x3               //[1,10]
       tmp_buf = tmp_buf0
   }
/* -------------------------------------------------------------*/
   {
       x00 = vmem(tmp_buf++#1)
       loop1(.L_depth1, rows)
   } {
       x10 = vmem(tmp_buf++#1)
       eob_count = add(rows, #-1)
   } {
       x20 = vmem(tmp_buf++#1)
   } {
       x30 = vmem(tmp_buf++#1)
   }
/* -------------------------------------------------------------*/
       .balign 32
.L_depth1:
   {
       x01 = vmem(tmp_buf++#1)
       p0 = cmp.eq(eob_count, #0)                //at end of block of in_depth flush pipe
       sum0.w = vadd(x00.w, vkappa.w) 
   } {
       if(p0) x01 = vzero                        //empty pipe at end
       x11 = vmem(tmp_buf++#1)
       sum1.w = vadd(x10.w, vkappa.w)
   } {
       if(p0) x11 = vzero                        //empty pipe at end
       x21 = vmem(tmp_buf++#1)
       sum2.w = vadd(x20.w, vkappa.w)
       d3 = #0
   } {
       if(p0) x21 = vzero                        //empty pipe at end
       x31 = vmem(tmp_buf++#1)
       offset = asl(radius, #2)                  //convert to ints radiux
   } {
       offset1 = #4                              //start from 1
       if(p0) x31 = vzero                        //empty pipe at end
       sum3.w = vadd(x30.w, vkappa.w)
       loop0(.L_win_sum, radius)                 //win is the window radious wn = 2N+1        
   }
/* -------------------------------------------------------------*/
       .balign 32
.L_win_sum:                                      //perform window sum for this level of depth
   {
       d0 = vlalign(x00, vzero0, offset)
       sum3.w = vadd(sum3.w, d3.w)
   } {
       sum0.w = vadd(sum0.w, d0.w)
       d0 = valign(x01, x00, offset1)
   } {
       sum0.w = vadd(sum0.w, d0.w)
       d1 = vlalign(x10, vzero1, offset)
   } {
       sum1.w = vadd(sum1.w, d1.w)
       d1 = valign(x11, x10, offset1)
   } {
       sum1.w = vadd(sum1.w, d1.w)
       d2 = vlalign(x20, vzero2, offset)
   } {
       sum2.w = vadd(sum2.w, d2.w)
       d2 = valign(x21, x20, offset1)
   } {
       sum2.w = vadd(sum2.w, d2.w)
       d3 = vlalign(x30, vzero3, offset)
       offset = add(offset, #-4)
   } {
       sum3.w = vadd(sum3.w, d3.w)
       d3 = valign(x31, x30, offset1)
       offset1 = add(offset1, #4)
   }:endloop0
/* -------------------------------------------------------------*/
   {
       eob_count = add(eob_count, #-1)
       vzero0 = x00
       x00 = x01
       r19 = #31
   } {
       vzero1 = x10
       x10 = x11
       sum3.w = vadd(sum3.w, d3.w)
       v31 = vsplat(r19)
   } {
       vzero2 = x20
       x20 = x21
       xe0.uw = vcl0(sum0.uw)                   //xe = Q6_R_cl0_R(sum);find leading bit
   } {
       vzero3 = x30
       x30 = x31
       xm0.w = vasl(sum0.w, xe0.w)       //xm=(sum<<xe)>>16; //overshift left to make 0-1
       xe0.w = vsub(v31.w, xe0.w)        //xe=31-xe; number is xm * 2^xe log of this xe + log(xm)
   } {
       xe1.uw = vcl0(sum1.uw)
   } {
       xm1.w = vasl(sum1.w, xe1.w)
       xe1.w = vsub(v31.w, xe1.w)
   } {
       xe2.uw = vcl0(sum2.uw)
   } {
       xm1xm0.h = vpacko(xm1.w, xm0.w)
       xm2.w = vasl(sum2.w, xe2.w)
       xe2.w = vsub(v31.w, xe2.w)
   } {
       xe3.uw = vcl0(sum3.uw)
   } {
       xm3.w = vasl(sum3.w, xe3.w)
       xe3.w = vsub(v31.w, xe3.w)
       r19 = #15
   } {
       xe0.w = vasl(xe0.w, r19)              //xe <<= 15
       r18 = ##0x78787878;                   //0111100s
   } {
       xe1.w = vasl(xe1.w, r19)              //
       xm3xm2.h = vpacko(xm3.w, xm2.w)       //part of log2
       v31 = vsplat(r18)
   } {
       xe2.w = vasl(xe2.w, r19)              //
       r18 = ##0x07ff07ff;
       tab = vmem(log_exp+#2)               //xm = log2tab[range][2];
   } {
       xe3.w = vasl(xe3.w, r19)              //
       range.b = vshuffo(xm3xm2.b, xm1xm0.b) 
       v29 = vsplat(r18)
   } {
       range = vand(range, v31)              //xm=log21px(xm); only do xm, exponent is a shift
       dx3dx2 = vand(xm3xm2, v29)            //deltax = (xm & 0x7ff);
       dx1dx0 = vand(xm1xm0, v29)
       c0 = #0
   } {
       r19 = #3
       dx3dx2.h = vadd(dx3dx2.h, dx3dx2.h)   //deltax = deltax+deltax;   //scale factor 2
   } {
       range.uh = vlsr(range.uh, r19)        //range = (xm >> 11) & 15;
   } {
       xm3xm2_xm1xm0.h = vlut16(range.b, tab.h, c0)
       tab = vmem(log_exp+#1)               //xm= mpyrsat(xm, deltax) + log2tab[range][1];
   } {
       dx1dx0.h = vadd(dx1dx0.h, dx1dx0.h)
   } {
       xm3xm2.h = vmpy(xm3xm2.h, dx3dx2.h):<<1:rnd:sat 
       coef1_coef0.h = vlut16(range.b, tab.h, c0)
   } {
       xm1xm0.h = vmpy(xm1xm0.h, dx1dx0.h):<<1:rnd:sat 
       xm3xm2.h = vadd(xm3xm2.h, coef1.h)
   } {
       xm1xm0.h = vadd(xm1xm0.h, coef0.h)
   } {
       xm3xm2.h = vmpy(xm3xm2.h, dx3dx2.h):<<1:rnd:sat 
       tab.tmp = vmem(log_exp+#0)            //xm= mpyrsat(xm, deltax) + log2tab[range][0];
       coef1_coef0.h = vlut16(range.b, tab.h, c0)
   } {
       xm1xm0.h = vmpy(xm1xm0.h, dx1dx0.h):<<1:rnd:sat 
       xm3xm2.h = vadd(xm3xm2.h, coef1.h)
   } {
       xm1xm0.h = vadd(xm1xm0.h, coef0.h)
   } {
       xm3_xm2 = vshuff(vzero, xm3xm2, cm2)
       r18 = #12
   } {
       xm1_xm0 = vshuff(vzero, xm1xm0, cm2)
       xe2 = vor(xm2, xe2)
       xe3 = vor(xm3, xe3)
       r19 = #15
   } {
       xe0 = vor(xm0, xe0)                   //x=(((int) xe)<<15)|(int)xm;reassemble xe + log(xm)
       xe1 = vor(xm1, xe1)
       v31 = vsplat(r18)
   } {
       xm2.w = vmpyo(xe2.w, vbeta.h):<<1:rnd:sat
   } {
       xm2.w = vsub(vsigma.w, xm2.w)
       xm3.w = vmpyo(xe3.w, vbeta.h):<<1:rnd:sat
   } {
       xm3.w = vsub(vsigma.w, xm3.w)    //x = sigma-x; subtract from correction factor 
       xm0.w = vmpyo(xe0.w, vbeta.h):<<1:rnd:sat//x=x*beta);//apply beta*log(sum) 16*32bit
   } {
       xm1.w = vmpyo(xe1.w, vbeta.h):<<1:rnd:sat
       xm0.w = vsub(vsigma.w, xm0.w)
       xe2.w = vasr(xm2.w, r19)
   } {
       xm1.w = vsub(vsigma.w, xm1.w)
       xe3.w = vasr(xm3.w, r19)
       xe2.w = vadd(xe2.w, v31.w)
   } {
       xe0.w = vasr(xm0.w, r19)
       xe3.w = vadd(xe3.w, v31.w)
   } {
       xe1.w = vasr(xm1.w, r19)
       xe0.w = vadd(xe0.w, v31.w)      //xe+12
       xm1xm0.h = vpacke(xm1.w, xm0.w)
   } {
       xe1.w = vadd(xe1.w, v31.w)
       xm3xm2.h = vpacke(xm3.w, xm2.w) //ym = exp2mx(x & 0x7fff); exponent of fraction xe is a shift left
   } {
       xe3_xe2 = vdeal(xe3, xe2, cm4) //ye = (x>>15) //get the final exponent
       r18 = ##0x78787878;                   //01111000
   } {
       xe1_xe0 = vdeal(xe1, xe0, cm4)        //get into right order
       range.b = vshuffo(xm3xm2.b, xm1xm0.b)
       v31 = vsplat(r18)
   } {
       range = vand(range, v31)
       r19 = #3
       r18 = ##0x07ff07ff;
   } {
       v31 = vsplat(r18)
   } {
       range.uh = vlsr(range.uh, r19)        //range = (xm >> 11) & 15;
       dx3dx2 = vand(xm3xm2, v31)            //deltax = (xm & 0x7ff);
       dx1dx0 = vand(xm1xm0, v31)
   } {
       tab.tmp = vmem(log_exp+#5)               //ym=  exp2tab[range][2];
       xm3xm2_xm1xm0.h = vlut16(range.b, tab.h, c0)
   } {
       tab.tmp = vmem(log_exp+#4)               //ym= mpyrsat(ym, deltax) + exp2tab[range][1];
       coef1_coef0.h = vlut16(range.b, tab.h, c0)
   } {
       xm3xm2.h = vmpy(xm3xm2.h, dx3dx2.h):<<1:rnd:sat 
   } {
       xm3xm2.h = vadd(xm3xm2.h, coef1.h)
       xm1xm0.h = vmpy(xm1xm0.h, dx1dx0.h):<<1:rnd:sat 
   } {
       xm1xm0.h = vadd(xm1xm0.h, coef0.h)
       tab.tmp = vmem(log_exp+#3)                 //ym= mpyrsat(ym, deltax) + exp2tab[range][0];
       coef1_coef0.h = vlut16(range.b, tab.h, c0)
   } {
       xm3xm2.h = vmpy(xm3xm2.h, dx3dx2.h):<<1:rnd:sat 
   } {
       xm1xm0.h = vmpy(xm1xm0.h, dx1dx0.h):<<1:rnd:sat 
       xm3xm2.h = vadd(xm3xm2.h, coef1.h)
       cm1 = #-1
   } {
       xm1xm0.h = vadd(xm1xm0.h, coef0.h)
   } {
       x3x2x1x0.tmp = vmem(in_ptr_o++M0)           //xm = ptr_x[128*i+j+w]+in_offset;//back into a real number
       x3x2_x1x0=vshuff(vzero,x3x2x1x0,cm1)        //shuffle in zeros
   } {
       x3x2.h = vadd(x3x2.h, vin_offset.h)         //
   } {
       x1x0.h = vadd(x1x0.h, vin_offset.h)         //
   } {
       xm3_xm2.w = vmpy(xm3xm2.h, x3x2.h)          //
   } {
       xm1_xm0.w = vmpy(xm1xm0.h, x1x0.h)          //y=xm*ym//apply lrn factor and output range value
   } {
       xm0.w = vasr(xm0.w, xe0.w)
   } {
       xm1.w = vasr(xm1.w, xe1.w)
   } {
       xm2.w = vasr(xm2.w, xe2.w)
   } {
       xm3.w = vasr(xm3.w, xe3.w)
       xm1xm0.h = vpacke(xm1.w, xm0.w)
   } {
   } {
       xm1xm0.h = vmpy(xm1xm0.h, recip.h):<<1:rnd:sat  //ym = (ym*recip+0x4000)>15
       xm3xm2.h = vpacke(xm3.w, xm2.w)
   } {
       xm1xm0.h = vsub(xm1xm0.h, vout_offset.h)    //produce the normalized output values in 0-255
   } {
       xm3xm2.h = vmpy(xm3xm2.h, recip.h):<<1:rnd:sat
   } {
       xm1xm0.h = vshuff(xm1xm0.h)    
   } {
       xm3xm2.h = vshuff(xm3xm2.h)   
   } {
       xm3xm2.h = vsub(xm3xm2.h, vout_offset.h)    //ym = (y >> (15+ye))-out_offset;
   } {
       x3x2x1x0.ub = vpack(xm3xm2.h, xm1xm0.h):sat //if(ym>255)ym=255; elsif(ym<0)ym=0;sat 8bits
       vmem(lrn_ptr++M0) = x3x2x1x0.new
   }:endloop1
/* -------------------------------------------------------------*/
   {
       in_ptr0 = add(in_ptr0, #128)
       lrn_ptr0 = add(lrn_ptr0, #128)
       p0 = cmp.eq(col_count, #0)
       if(!p0.new) jump:t .L_width
   } {
       in_ptr00 = add(in_ptr00, next_logcl_row)     //next logical row-(in_depth/32)*width*32
       lrn_ptr00 = add(lrn_ptr00, next_logcl_row)
       p0 = cmp.eq(in_height, #0)
       if(!p0.new) jump:t .L_height
   }
/* -------------------------------------------------------------*/
   {
       r17:16 = memd(sp+#0)
       r19:18 = memd(sp+#8)
   } {
       r21:20 = memd(sp+#16) 
       r23:22 = memd(sp+#24) 
   } {
       r25:24 = memd(sp+#32) 
       r27:26 = memd(sp+#40) 
   }  {
       dealloc_return
   }
.L_end:
/* -------------------------------------------------------------*/
      .size lrn_d32_hvx, .L_end-lrn_d32_hvx

/*=============================================================================*/
