/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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

//  L2Normalize hvx optimized. error is within 4 steps
//  Independently normalizes each 1-D slice along the given axis. Axis is depth by default
//  for each axis slice, output = x / sqrt(sum(x*x))
//  the op quantizes the output tensor to [-1,1]
//  3..4 inputs:
//      0           input data       (uint8_t)
//      1           input min val    (scalar float)
//      2           input max val    (scalar float)
//      3(optional) input axis index (int32_t)
//  3 output:
//      0   output data      (uint8_t)
//      1   onput min val    (scalar float)
//      2   onput max val    (scalar float)

#include <nn_graph.h>
#include <math.h>
#include "nn_axis.h"
#include "quantize.h"
#include "hvx_inlines.h"
#include "hvx_mathops.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "nn_reduce_utils.h"

#define OP_L2Norm_IN_NUM    3
#define OP_L2Norm_IN_NUM_WITH_OPTIONAL    4
#define OP_L2Norm_OUT_NUM   3
#define AXIS_DEPTH_IDX      3
#define IN_DATA_IDX         0
#define IN_MIN_IDX          1
#define IN_MAX_IDX          2
#define IN_AXIS_IDX         3
#define OUT_DATA_IDX        0
#define OUT_MIN_IDX         1
#define OUT_MAX_IDX         2
#define OUT_MIN             -1
#define OUT_MAX             1

//process one slice of data along an axis. If the axis is w, process w x d slice. If the axis is h, process h x d slice.
//stride is inner stride
static inline void hvx_l2norm_in_cols(struct nn_graph *nn, const uint8_t* in_data, uint8_t in_0_offset, int dim_value, int depth, int stride, uint8_t* out_data,
                         uint8_t out_0_off) {
    int slices = (depth+127)/128u;		// Slices are devided by 128 bytes along depth.
                                        // One slice has dim (b,h, or w) * 128 or dim (b,h, or w) * (depth%128) bytes
    HVX_Vector vin_offset = q6op_Vb_vsplat_R(in_0_offset);
    HVX_Vector vin_128 = q6op_Vb_vsplat_R(0x80);    //offset
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector v_15 = Q6_V_vsplat_R(Q6_R_combine_RlRl(0,15));
    HVX_Vector v_1 = Q6_V_vsplat_R(1);

    int row_increment = 0;
    for( int islc = 0; islc < slices; islc++) { //traverse along depth
        int slc_cols = min_i32( depth-128*islc, 128);	// # of dim_value in this block
        int col_inc = islc*128;
        HVX_Vector sum_lo_lo = Q6_V_vzero();   //keeps the sum across the inner loop
        HVX_Vector sum_lo_hi = Q6_V_vzero();
        HVX_Vector sum_hi_lo = Q6_V_vzero();
        HVX_Vector sum_hi_hi = Q6_V_vzero();

        for(int rowbase = 0; rowbase < dim_value; ++rowbase){
            row_increment = rowbase*stride + col_inc;
            HVX_Vector vin = q6op_V_vldu_A((HVX_Vector const*)(in_data + row_increment));
            vin = Q6_Vub_vabsdiff_VubVub(vin, vin_offset);
            HVX_VectorPair vin_square = Q6_Wuh_vmpy_VubVub(vin,vin);   //lo = x

            HVX_VectorPair tmp_sum_lo = Q6_Wh_vshuffoe_VhVh(zero, Q6_V_lo_W(vin_square));   //extend half-word to word,even
            HVX_VectorPair tmp_sum_hi = Q6_Wh_vshuffoe_VhVh(zero, Q6_V_hi_W(vin_square));   //odd

            sum_lo_lo = Q6_Vw_vadd_VwVw_sat(Q6_V_lo_W(tmp_sum_lo),  sum_lo_lo); //keep the sum of even elements along depth
            sum_lo_hi = Q6_Vw_vadd_VwVw_sat(Q6_V_hi_W(tmp_sum_lo),  sum_lo_hi);
            sum_hi_lo = Q6_Vw_vadd_VwVw_sat(Q6_V_lo_W(tmp_sum_hi),  sum_hi_lo); //keep the sum of odd elements along depth
            sum_hi_hi = Q6_Vw_vadd_VwVw_sat(Q6_V_hi_W(tmp_sum_hi),  sum_hi_hi);
        }
        //Get reciprocal square root
        HVX_Vector shift_lo_lo = Q6_Vuw_vcl0_Vuw(sum_lo_lo);//count leading 0s
        HVX_Vector shift_lo_hi = Q6_Vuw_vcl0_Vuw(sum_lo_hi);
        HVX_Vector shift_hi_lo = Q6_Vuw_vcl0_Vuw(sum_hi_lo);
        HVX_Vector shift_hi_hi = Q6_Vuw_vcl0_Vuw(sum_hi_hi);

        //left shift by the amount of leading 0s to give the values 16 fractional bits, which ensure the values > 0x4000
        shift_lo_lo = Q6_Vw_vsub_VwVw(shift_lo_lo, Q6_V_vsplat_R(1));//leave one bit for the sign
        shift_lo_hi = Q6_Vw_vsub_VwVw(shift_lo_hi, Q6_V_vsplat_R(1));
        shift_hi_lo = Q6_Vw_vsub_VwVw(shift_hi_lo, Q6_V_vsplat_R(1));
        shift_hi_hi = Q6_Vw_vsub_VwVw(shift_hi_hi, Q6_V_vsplat_R(1));

        sum_lo_lo = Q6_Vw_vasl_VwVw(sum_lo_lo, shift_lo_lo);    //<< shift_lo_lo
        sum_lo_hi = Q6_Vw_vasl_VwVw(sum_lo_hi, shift_lo_hi);
        sum_hi_lo = Q6_Vw_vasl_VwVw(sum_hi_lo, shift_hi_lo);
        sum_hi_hi = Q6_Vw_vasl_VwVw(sum_hi_hi, shift_hi_hi);

        shift_lo_lo = Q6_Vw_vsub_VwVw(shift_lo_lo, v_15);   //(leading 0s -1 -15 )/2
        shift_lo_hi = Q6_Vw_vsub_VwVw(shift_lo_hi, v_15);
        shift_hi_lo = Q6_Vw_vsub_VwVw(shift_hi_lo, v_15);
        shift_hi_hi = Q6_Vw_vsub_VwVw(shift_hi_hi, v_15);

        HVX_Vector select = Q6_V_vand_VV(shift_lo_lo, v_1);  //e.g. word 0001
        select = Q6_Vw_vaslacc_VwVwR(select, select, 8);    //0010+0001 = 0011
        select = Q6_Vw_vaslacc_VwVwR(select, select, 16);
        HVX_Vector rsqrt_lo_lo = rsqrt_newton_hvx(nn,sum_lo_lo, select);//31-bit scale factor

        select = Q6_V_vand_VV(shift_lo_hi, v_1);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 8);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 16);
        HVX_Vector rsqrt_lo_hi = rsqrt_newton_hvx(nn,sum_lo_hi, select);//31-bit scale factor

        select = Q6_V_vand_VV(shift_hi_lo, v_1);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 8);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 16);
        HVX_Vector rsqrt_hi_lo = rsqrt_newton_hvx(nn,sum_hi_lo, select);

        select = Q6_V_vand_VV(shift_hi_hi, v_1);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 8);
        select = Q6_Vw_vaslacc_VwVwR(select, select, 16);
        HVX_Vector rsqrt_hi_hi = rsqrt_newton_hvx(nn,sum_hi_hi, select);

        rsqrt_lo_lo = Q6_Vw_vasr_VwR(rsqrt_lo_lo, 16);
        rsqrt_lo_hi = Q6_Vw_vasr_VwR(rsqrt_lo_hi, 16);

        HVX_VectorPair shuff = Q6_W_vshuff_VVR(rsqrt_lo_hi, rsqrt_lo_lo, -4);
        HVX_Vector rsqrt_lo = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

        rsqrt_hi_lo = Q6_Vw_vasr_VwR(rsqrt_hi_lo, 16);
        rsqrt_hi_hi = Q6_Vw_vasr_VwR(rsqrt_hi_hi, 16);
        shuff = Q6_W_vshuff_VVR(rsqrt_hi_hi, rsqrt_hi_lo, -4);
        HVX_Vector rsqrt_hi = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

        HVX_Vector rsh = Q6_V_vsplat_R(8);//31(scale factor)
                                        // -7(keep 7 bits)
                                        // -16(pack the result to 16b)
                                        // + floor((leading 0s -1 -15 )/2) later

        shift_lo_lo = Q6_Vw_vadd_VwVw(shift_lo_lo, v_1);//round for >> 1
        shift_lo_hi = Q6_Vw_vadd_VwVw(shift_lo_hi, v_1);
        shift_hi_lo = Q6_Vw_vadd_VwVw(shift_hi_lo, v_1);
        shift_hi_hi = Q6_Vw_vadd_VwVw(shift_hi_hi, v_1);

        shift_lo_lo = Q6_Vw_vasr_VwR(shift_lo_lo, 1);   //since << shift_lo_lo then get reciprocal sqrt, only >> shift_lo_lo/2 to scale back
        shift_lo_hi = Q6_Vw_vasr_VwR(shift_lo_hi, 1);
        shift_hi_lo = Q6_Vw_vasr_VwR(shift_hi_lo, 1);
        shift_hi_hi = Q6_Vw_vasr_VwR(shift_hi_hi, 1);

        shift_lo_lo = Q6_Vw_vsub_VwVw(rsh, shift_lo_lo);//(leading 0s -1 -15 )/2
        shift_lo_hi = Q6_Vw_vsub_VwVw(rsh, shift_lo_hi);
        shift_hi_lo = Q6_Vw_vsub_VwVw(rsh, shift_hi_lo);
        shift_hi_hi = Q6_Vw_vsub_VwVw(rsh, shift_hi_hi);

        for(int rowbase = 0; rowbase < dim_value; ++rowbase){
            row_increment = rowbase*stride + col_inc;
            HVX_Vector vin = q6op_V_vldu_A((HVX_Vector const*)(in_data + row_increment));
            HVX_VectorPair diff_pair = Q6_Wh_vsub_VubVub(vin, vin_offset);

            HVX_VectorPair mul_diff_lo = Q6_Ww_vmpy_VhVh( Q6_V_lo_W(diff_pair), rsqrt_lo);  //(tmp_out * frac)
            HVX_VectorPair mul_diff_hi = Q6_Ww_vmpy_VhVh( Q6_V_hi_W(diff_pair), rsqrt_hi);

            HVX_Vector rsh_lo = Q6_Vw_vasr_VwVw(Q6_V_lo_W(mul_diff_lo), shift_lo_lo);
            HVX_Vector rsh_hi = Q6_Vw_vasr_VwVw(Q6_V_hi_W(mul_diff_lo), shift_lo_hi);
            shuff = Q6_W_vshuff_VVR(rsh_hi, rsh_lo, -4);
            HVX_Vector res_lo = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            rsh_lo = Q6_Vw_vasr_VwVw(Q6_V_lo_W(mul_diff_hi), shift_hi_lo);
            rsh_hi = Q6_Vw_vasr_VwVw(Q6_V_hi_W(mul_diff_hi), shift_hi_hi);
            shuff = Q6_W_vshuff_VVR(rsh_hi, rsh_lo, -4);
            HVX_Vector res_hi = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            shuff = Q6_W_vshuff_VVR(res_hi, res_lo, -2); //back to the original order
            HVX_Vector res = Q6_Vb_vpack_VhVh_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            res = Q6_Vb_vadd_VbVb(vin_128, res);    //map [-128,128] to [0,255]

            q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)out_data+row_increment), slc_cols, res);
        }
    }
}

//along depth
static inline void hvx_l2norm_in_rows(struct nn_graph *nn, const uint8_t* in_data, uint8_t in_0_offset, int rows, int dim_val, int stride, uint8_t* out_data,
                        uint8_t out_0_off) {

    HVX_VectorPred lastmask = Q6_Q_vsetq_R(dim_val%128);
    int vecs_across = (dim_val+127)/128u;
    int ilast = dim_val/128u;

    //init for getting the sum of each depth
    int32_t squared_l2_norm = 0;
    int row_increment = 0;

    for(int rowno = 0; rowno < rows; ++rowno) {  //increase along width
        row_increment = rowno*stride;
        HVX_Vector const *inrow = (HVX_Vector const *)(in_data + row_increment);

        HVX_Vector sum = Q6_V_vzero();//Q6_V_vsplat_R(0);
        HVX_Vector vin_offset = q6op_Vb_vsplat_R(in_0_offset);
        HVX_Vector vin_128 = q6op_Vb_vsplat_R(0x80);

        for( int i =0; i < vecs_across; i++){ //traverse along depth until getting the sum of the whole depth row
            HVX_Vector vin = q6op_V_vldu_A(inrow+i);
            vin = Q6_Vub_vabsdiff_VubVub(vin, vin_offset);

            if( i == ilast){	// blank out the extra bytes
                vin = q6op_V_vand_QV(lastmask, vin);
            }

            HVX_VectorPair vin_square = Q6_Wuh_vmpy_VubVub(vin,vin);   //lo = x

            HVX_VectorPair sum_pair = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(vin_square), Q6_V_hi_W(vin_square)); //vin0^2 even + vin0^2 odd.
                                                                                // has depth/2 elems in the vector pair
            HVX_Vector tmp_sum = Q6_Vw_vadd_VwVw_sat(Q6_V_lo_W(sum_pair),  Q6_V_hi_W(sum_pair)); //keep the sum in one vector for each row
            sum = Q6_Vw_vadd_VwVw_sat(sum, tmp_sum);
        }//traverse depth

        sum = compute_reduced_sum(sum); //sum of the vector
        squared_l2_norm = *(uint32_t *)&sum;
        const float l2_norm = sqrtf(squared_l2_norm);

        // x / l2_norm = ( x * 2^15 / l2_norm_mantissa ) >> (8+rsh)
        int rsh = floor_log2(l2_norm);    //get the bits of the exponent of l2_norm.
        if(0.0 == l2_norm) {
            HVX_Vector out_offset = q6op_Vb_vsplat_R(out_0_off);
            q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)out_data+row_increment), dim_val, out_offset);   //set row to 0
            logmsg(nn, 1, "op_l2normalize_8 elements in row %d are 0s", rowno);
            continue;
        }

        int frac = ((unsigned)32768/ (unsigned) l2_norm) << rsh;  //2^15/l2_norm_mantissa
        frac = Q6_R_combine_RlRl(frac, frac);

        for( int i =0; i < vecs_across; i++) { //traverse along depth until getting the sum of the whole depth row
            HVX_Vector vin = q6op_V_vldu_A(inrow+i);

            //tmp_out = (int)in_data[tmp_idx] - (int)in_0_offset ;  // >=0 , <2^8
            HVX_VectorPair diff_pair = Q6_Wh_vsub_VubVub(vin, vin_offset);

            HVX_VectorPair mul_diff_lo = Q6_Ww_vmpy_VhRh( Q6_V_lo_W(diff_pair), frac);  //(tmp_out * frac)
            HVX_VectorPair mul_diff_hi = Q6_Ww_vmpy_VhRh( Q6_V_hi_W(diff_pair), frac);

            int tmp = rsh +8;
            HVX_Vector shift_hi = Q6_Vw_vasr_VwR(Q6_V_hi_W(mul_diff_lo), tmp);
            HVX_Vector shift_lo = Q6_Vw_vasr_VwR(Q6_V_lo_W(mul_diff_lo), tmp);
            HVX_VectorPair shuff = Q6_W_vshuff_VVR(shift_hi, shift_lo, -4);
            HVX_Vector shift_rsh_lo = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            shift_hi = Q6_Vw_vasr_VwR(Q6_V_hi_W(mul_diff_hi), tmp);
            shift_lo = Q6_Vw_vasr_VwR(Q6_V_lo_W(mul_diff_hi), tmp);
            shuff = Q6_W_vshuff_VVR(shift_hi, shift_lo, -4);
            HVX_Vector shift_rsh_hi = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            shuff = Q6_W_vshuff_VVR(shift_rsh_hi, shift_rsh_lo, -2); //back to the original order
            HVX_Vector res = Q6_Vb_vpack_VhVh_sat(Q6_V_hi_W(shuff), Q6_V_lo_W(shuff));

            res = Q6_Vb_vadd_VbVb(vin_128, res);    //map [-128,128] to [0,255]

            if( i == ilast){	// blank out the extra bytes
                q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)out_data+row_increment+i*128), dim_val%128u, res);
            }
            else {
                q6op_vstu_variable_ARV((HVX_Vector *) ((uint8_t*)out_data+row_increment+i*128), 128, res);
            }
        }
    }
}

struct l2norm_runstate {
    struct l2norm_info * info;
    nn_sem_t done_sem;
};

struct l2norm_info {
    uint8_t * out_data;
    uint8_t * in_data;
    int32_t axis;
    uint8_t in_0_offset;
    int32_t b;
    int32_t h;
    int32_t w;
    int32_t d;
    int32_t data_stride;
    uint8_t out_0_off;
};

static void l2norm_hvx(struct nn_graph *nn,  void * rstpv) {

    struct l2norm_runstate *rstp = (struct l2norm_runstate *)rstpv;
    struct l2norm_info const * info = rstp->info;

    uint8_t * in_data = info->in_data;
    uint8_t * out_data = info->out_data;
    int32_t axis = info->axis;
    uint8_t in_0_offset = info->in_0_offset;
    int32_t b = info->b;
    int32_t h = info->h;
    int32_t w = info->w;
    int32_t d = info->d;
    int32_t data_stride = info->data_stride;
    uint8_t out_0_off = info->out_0_off;

    if( AXIS_DEPTH_IDX == axis ) {
        hvx_l2norm_in_rows(nn, in_data, in_0_offset, b*h*w, d, d, out_data, out_0_off);
    }else {
        int32_t increment = 0;  //increment of the data pointer
        int32_t increment_h = (h-1)*w*d + d;    //height increment of across a batch block
        int32_t outer_loop = 0;
        int32_t inner_loop = 0;
        switch(axis) {
            case 0: //batch
                outer_loop = w * h;
                inner_loop = b;
                increment = d;
                break;
            case 1: //height
                outer_loop = w * b;
                inner_loop = h;
                increment = d;
                break;
            case 2: //width
                outer_loop = b * h;
                inner_loop = w;
                increment = w * d;
                break;
            default:
                break;
        }
        int32_t data_inc = 0;
        for(int outer = 0; outer < outer_loop; ++outer) {

            hvx_l2norm_in_cols(nn,in_data+data_inc, in_0_offset, inner_loop, d, data_stride, out_data+data_inc, out_0_off);
            if((axis == 1) && ((outer+1)%w == 0)) {//height
                data_inc += increment_h;
            }
            else {
                data_inc += increment;
            }
        }
    }
    nn_sem_post(& rstp->done_sem);
}

static int l2normalize_execute_8_fast(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_data_tensor = self->inputs[IN_DATA_IDX];
    const struct tensor *in_min_tensor = self->inputs[IN_MIN_IDX];
    const struct tensor *in_max_tensor = self->inputs[IN_MAX_IDX];
    struct tensor *out_data_tensor = self->outputs[OUT_DATA_IDX];
    struct tensor *out_min_tensor = self->outputs[OUT_MIN_IDX];
    struct tensor *out_max_tensor = self->outputs[OUT_MAX_IDX];
    const struct shape in_shape = in_data_tensor->shape;

    const float in_min = tensor_get_float(in_min_tensor,0);
    const float in_max = tensor_get_float(in_max_tensor,0);
    uint8_t* out_data = out_data_tensor->data;

    int32_t in_axis = AXIS_DEPTH_IDX;
    if (self->n_inputs == OP_L2Norm_IN_NUM_WITH_OPTIONAL) {
        in_axis = tensor_get_int32(self->inputs[IN_AXIS_IDX],0);
        handle_negative_axes(nn, &in_axis, 1);
    }

    if (tensor_out_prepare_normal_fromshape(out_data_tensor, &in_shape, NN_TYPE_UINT8) !=0) return errlog(nn,"op_l2normalize out too small");

    int inner_stride = 1;
    for (int i = AXIS_DEPTH_IDX; i > in_axis; i--)
    {
        inner_stride *= in_shape.dimension[i];
    }

    uint8_t in_0_offset = quantize_uint8(0.0f, in_min, in_max); //quantized val of 0.0f
    float out_min = OUT_MIN;
    float out_max = OUT_MAX;

    float out_scale = (float)(out_max - out_min)/255.0;
    uint8_t out_0_off = saturate_u8((int32_t)((float)(0-out_min)/out_scale));

    tensor_set_single_float( out_min_tensor, out_min);
    tensor_set_single_float( out_max_tensor, out_max);

    struct l2norm_info info;
    info.out_data = out_data;
    info.in_data = in_data_tensor->data;
    info.in_0_offset = in_0_offset;
    info.b = in_shape.batches;
    info.h = in_shape.height;
    info.w = in_shape.width;
    info.d = in_shape.depth;
    info.data_stride = inner_stride;
    info.out_0_off = out_0_off;
    info.axis = in_axis;

    struct l2norm_runstate rst;
    rst.info = &info;

    nn_sem_init( &rst.done_sem, 0);
    nn_os_work_for_vector(nn,  l2norm_hvx, &rst );
    nn_sem_wait( &rst.done_sem);

    return 0;
}

struct nn_node_ops nn_ops_for_L2Normalize_8 = {
    .execute = l2normalize_execute_8_fast,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT_RANGE(OP_L2Norm_IN_NUM, OP_L2Norm_IN_NUM_WITH_OPTIONAL),
    .n_outputs = NN_IOCOUNT(OP_L2Norm_OUT_NUM),
};


