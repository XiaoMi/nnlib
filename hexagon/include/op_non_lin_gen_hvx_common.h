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

#ifndef NN_OP_NON_LIN_GEN_HVX_COMMON_H
#define NN_OP_NON_LIN_GEN_HVX_COMMON_H

#include <op_non_lin_common.h>

#define vmem(A)  *((HVX_Vector*)(A))

#define Q6_VB_VADD_VBVUB_SAT(VCOEF,VOUT,VTMPI0,VTMPI1,VZERO,VUNITY,WCOEF,WTEMP0,WTEMP1,VPRED) \
        WCOEF = Q6_Wh_vmpy_VubVb(VCOEF,VUNITY); \
        WTEMP1 = Q6_Wh_vmpy_VubVb(VOUT,VUNITY); \
        WTEMP1 = Q6_Wh_vadd_WhWh_sat(WCOEF,WTEMP1); \
        WTEMP0 = Q6_Wh_vmpy_VubVb(VZERO,VUNITY); \
        VPRED  = Q6_Q_vcmp_gt_VhVh(Q6_V_lo_W(WTEMP0),Q6_V_lo_W(WTEMP1)); \
        VTMPI0 = Q6_V_vmux_QVV(VPRED, Q6_V_lo_W(WTEMP0),Q6_V_lo_W(WTEMP1)); \
        VPRED  = Q6_Q_vcmp_gt_VhVh(Q6_V_hi_W(WTEMP0),Q6_V_hi_W(WTEMP1)); \
        VTMPI1 = Q6_V_vmux_QVV(VPRED, Q6_V_hi_W(WTEMP0),Q6_V_hi_W(WTEMP1)); \
        VOUT   = Q6_Vub_vasr_VhVhR_rnd_sat(VTMPI1,VTMPI0,0); \
        WTEMP1 = Q6_Wh_vmpy_VubVb(VOUT,VUNITY); \
        VTMPI0 = Q6_V_lo_W(WTEMP1); \
        VTMPI1 = Q6_V_hi_W(WTEMP1); \
        VOUT   = Q6_Vub_vasr_VhVhR_rnd_sat(VTMPI1,VTMPI0,0);

static int qnonlinear_execute_i(unsigned char *ptr_y, const unsigned char *ptr_x, int numbytes, const signed char *ptr_non_lin)
{
    //NOTE: nonlinear function minmax approximation with look-up table using linear polynomial app (polynomial order 1) 
    int bytecount;
    int rngmask, dltmask, dltmult;
    HVX_VectorPred vpred_lim;
    HVX_Vector *ptr_vx = (HVX_Vector *)ptr_x;
    HVX_Vector *ptr_vy = (HVX_Vector *)ptr_y;
    HVX_Vector vzero, vunity;
#ifndef OUT_RNG_FLOAT_ZERO_TO_POSITIVE
    HVX_Vector voffset;
#else
    HVX_Vector vdouble;
#endif
    HVX_Vector vrngmask, vdltmask, vdltmult;
    HVX_Vector vrngidx, vtmpi0, vtmpi1;
    HVX_Vector vin, vdeltax, vout;
    HVX_VectorPair wtemp0, wtemp1, wcoef; 
    HVX_Vector vcoef_polyorder0, vcoef_polyorder1, vlut_polyorder0, vlut_polyorder1;
    
    vzero = Q6_V_vsplat_R(0x00000000);
    vunity = Q6_V_vsplat_R(0x01010101);
    
#ifndef OUT_RNG_FLOAT_ZERO_TO_POSITIVE
    voffset = Q6_V_vsplat_R(0x80808080);
#else
    vdouble = Q6_V_vsplat_R(0x02020202);
#endif
    
    // vrngmask example  : 0xf0f0f0f0 (int bits b7-b4) or 0x78787878 (int bits b6-b3)
    rngmask = ((RNGIDX_MASK<<24) & 0xFF000000) | ((RNGIDX_MASK<<16) & 0x00FF0000) | ((RNGIDX_MASK<<8) & 0x0000FF00) | (RNGIDX_MASK & 0x000000FF);
    vrngmask = Q6_V_vsplat_R(rngmask);
    
    // vdltmask example  : 0x0f0f0f0f (frac bits b3-b0) or 0x07070707 (frac bits b2-b0)
    dltmask = ((DELTAX_MASK<<24) & 0xFF000000) | ((DELTAX_MASK<<16) & 0x00FF0000) | ((DELTAX_MASK<<8) & 0x0000FF00) | (DELTAX_MASK & 0x000000FF);
    vdltmask = Q6_V_vsplat_R(dltmask);
    
    // vdltmult example  : 0x04040404 (frac shiftleft=2 => mult=2^2)
    dltmult = ((DELTAX_MULT<<24) & 0xFF000000) | ((DELTAX_MULT<<16) & 0x00FF0000) | ((DELTAX_MULT<<8) & 0x0000FF00) | (DELTAX_MULT & 0x000000FF);
    vdltmult = Q6_V_vsplat_R(dltmult);
    
    // set lut pointers based on polynomial n_order
    vlut_polyorder0 = vmem(ptr_non_lin);
    ptr_non_lin += ALIGN_SIZE;
    vlut_polyorder1 = vmem(ptr_non_lin);
    ptr_non_lin += ALIGN_SIZE;
    
    for (bytecount = 0; bytecount < numbytes; bytecount += ALIGN_SIZE)
    {
        // load input x
        vin = *ptr_vx++;
        
        // rngidx = ((x & rngidx_mask) >> rngidx_rshift)
        vtmpi0 = Q6_V_vand_VV(vin, vrngmask); // (x & rngidx_mask) 
        vrngidx = Q6_Vuh_vlsr_VuhR(vtmpi0, RNGIDX_RSHIFT); // ((x & rngidx_mask) >> rngidx_rshift)
        
        // deltax = (x & deltax_mask) * (1 << deltax_lshift) = ((x & deltax_mask) << deltax_lshift)
        vtmpi0 = Q6_V_vand_VV(vin, vdltmask); // (x & deltax_mask)
        wtemp0 = Q6_Wh_vmpy_VubVb(vtmpi0, vdltmult); // (x & deltax_mask) * (1 << deltax_lshift)
        vdeltax = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(wtemp0),Q6_V_lo_W(wtemp0),0); // ((x & deltax_mask) << deltax_lshift)
        
        // read lut_non_lin
        vcoef_polyorder0 = Q6_Vb_vlut32_VbVbR(vrngidx, vlut_polyorder0, 0);
        vcoef_polyorder1 = Q6_Vb_vlut32_VbVbR(vrngidx, vlut_polyorder1, 0);
        
        // compute output using lut_non_lin & rngidx & deltax
        // init y =                    lut_non_lin[((ORDER+1)*rngidx)+(ORDER)];
        // loop y = mpyrsat(y, dltx) + lut_non_lin[((ORDER+1)*rngidx)+(ORDER-loopcount)];
        wtemp0 = Q6_Wh_vmpy_VubVb(vdeltax,vcoef_polyorder1);
        vout = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(wtemp0),Q6_V_lo_W(wtemp0),7);
        Q6_VB_VADD_VBVUB_SAT(vcoef_polyorder0,vout,vtmpi0,vtmpi1,vzero,vunity,wcoef,wtemp0,wtemp1,vpred_lim)
        
#ifndef OUT_RNG_FLOAT_ZERO_TO_POSITIVE
        // offset output y to convert from (-127,127) signed char to (0,255) unsigned char
        vout = Q6_Vb_vadd_VbVb(vout,voffset);
#else
        // double output y to convert from (0,127) signed char to (0,255) unsigned char
        wtemp1 = Q6_Wh_vmpy_VubVb(vout,vdouble);
        vout   = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(wtemp1),Q6_V_lo_W(wtemp1),0);
#endif
        
        // store output y
        *ptr_vy++ = vout;
    }
    
    return 0;
}

#endif
