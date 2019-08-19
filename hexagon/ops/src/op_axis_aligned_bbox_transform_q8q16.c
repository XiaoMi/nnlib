
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
#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>
#include "hvx_inlines.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains an axis-aligned bbox tranform op
 */


#define BYTE_VECTOR_SIZE 128
#define HALFWORD_VECTOR_SIZE 64
#define WORD_VECTOR_SIZE 32
#define UNIQUE_BYTE_VALUES 256
#define MAX_HALFWORD 65535
#define HALFWORD_BITS 16

#define FLT_ACCURACY_SHIFT 7
#define EXP_ACCURACY_SHIFT 6

#define ROI_LENGTH 4
#define HVX_BOXES_CHUNK_SIZE 256

#define BOX_DIM_STEP 0.125f


/**
 * Box descriptor that uses lower-left and upper-right vertex coordinates to
 * define a box.
 */
typedef struct {
    int32_t x1, y1, x2, y2;
} box_encoding_corner;

/**
 * Box descriptor that uses the width, height and center coordinates to
 * define a box.
 */
typedef struct {
    int32_t w, h, x, y;
} box_encoding_center;

/**
 * Parameters for HVX-optimized path.
 */
struct tdata {
    uint16_t* boxes;
    uint8_t* deltas;
    float deltas_min;
    float deltas_max;
    uint32_t num_boxes;
    uint32_t num_classes;
    uint16_t* out_data;
    nn_sem_t donesem;
};

/**
 * Creates a map from quantized values to floating point exponential values.
 * @param min Quantization minimum.
 * @param max Quantization maximum.
 * @param output An array of size 256. This parameter is filled by the function
 * in such a way that the value of the index is dequantized, applied to e^x
 * and the result is placed at that index's location in the array.
 */
static void get_exp_map_f(float min, float max, float *output){
    float step = (max - min) / 255.f;
    float float_val = min;

    for(int i = 0; i < UNIQUE_BYTE_VALUES; i++){
        output[i] = expf(float_val);
        float_val += step;
    }
}

/**
 * Creates a map from quantized values to floating point values.
 * @param min Quantization minimum.
 * @param max Quantization maximum.
 * @param output An array of size 256. This parameter is filled by the function
 * in such a way that the value of the index is dequantized and placed at that
 * index's location in the array.
 */
static void get_float_map_f(float min, float max, float *output){
    float step = (max - min) / 255.f;
    float float_val = min;

    for(int i = 0; i < UNIQUE_BYTE_VALUES; i++){
        output[i] = float_val;
        float_val += step;
    }
}

/**
 * De-quantizes all unique quantized values, applies the floating point values
 * to e^x, multiplies the result by a constant, converts it to uint16 and
 * stores the high and low bytes into seperate maps.
 * @param min Quantization minimum.
 * @param max Quantization maximum.
 * @param bytes1 Low-byte map.
 * @param bytes2 High-byte map.
 */
static void get_exp_maps_u16(float min, float max, unsigned char *bytes1, unsigned char *bytes2){
    float step = (max - min) / 255.f;
    float float_val = min;
    const float accuracy_multiplier = 1 << EXP_ACCURACY_SHIFT;

    for(int i = 0; i < UNIQUE_BYTE_VALUES; i++){
        uint16_t halfword = round(expf(float_val) * accuracy_multiplier);
        bytes1[i] = (halfword&0xFF);
        bytes2[i] = ((halfword>>8)&0xFF);
        float_val += step;
    }
}

/**
 * De-quantizes all unique quantized values, multiplies the result by a constant,
 * converts it to int16 and stores the high and low bytes into seperate maps.
 * @param min Quantization minimum.
 * @param max Quantization maximum.
 * @param bytes1 Low-byte map.
 * @param bytes2 High-byte map.
 */
static void get_float_maps_i16(float min, float max, unsigned char *bytes1, unsigned char *bytes2){
    float step = (max - min) / 255.f;
    float float_val = min;
    const float accuracy_multiplier = 1 << FLT_ACCURACY_SHIFT;

    for(int i = 0; i < UNIQUE_BYTE_VALUES; i++){
        int16_t halfword = round(float_val * accuracy_multiplier);
        bytes1[i] = (halfword&0xFF);
        bytes2[i] = ((halfword>>8)&0xFF);
        float_val += step;
    }
}

/**
 * Maps 8-bit values in an HVX vector using the given map.
 * @param vin Input values.
 * @param vout Mapped values.
 * @param map_data The map.
 */
static void map_values(HVX_Vector *vin, HVX_Vector *vout, unsigned char* map_data)
{
    HVX_Vector luta = *(HVX_Vector *) map_data;
    HVX_Vector lutb = *(HVX_Vector *) & map_data[BYTE_VECTOR_SIZE];
    HVX_Vector lut0 = Q6_Vb_vshuff_Vb(luta);
    HVX_Vector lut1 = Q6_Vb_vshuff_Vb(lutb);

    *vout = q6op_Vb_vlut32_VbVbI(*vin, lut0, 0);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut0, 1);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut0, 2);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut0, 3);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut1, 4);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut1, 5);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut1, 6);
    *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, *vin, lut1, 7);
}

/**
 * Reference implementation for the transformation step.
 * @param nn NN graph reference.
 * @param boxes Box coordinates tensor.
 * @param deltas_min Quantization min for deltas.
 * @param deltas_max Quantization max for deltas.
 * @param num_classes Number of classes.
 * @param boxes_data_end Reference to the end of boxes data.
 * @param deltas Delta values.
 * @param out_data Output buffer.
 * @return 0 if successful, error code otherwise.
 */
static int axis_aligned_bbox_transform_execute_ref(struct nn_graph *nn,
                                                   uint16_t* boxes, float deltas_min, float deltas_max, uint32_t num_classes,
                                                   const uint16_t* boxes_data_end, const uint8_t* deltas, uint16_t* out_data)
{
    float exp_map[UNIQUE_BYTE_VALUES];
    get_exp_map_f(deltas_min, deltas_max, exp_map);

    float float_map[UNIQUE_BYTE_VALUES];
    get_float_map_f(deltas_min, deltas_max, float_map);

    while (boxes < boxes_data_end) {

        if (boxes[0] > boxes[2]) return errlog(nn,"boxes[0] is not less than boxes[2]");
        if (boxes[1] > boxes[3]) return errlog(nn,"boxes[1] is not less than boxes[3]");

        box_encoding_center roi_before;
        roi_before.w = boxes[2] - boxes[0];
        roi_before.h = boxes[3] - boxes[1];
        roi_before.x = (boxes[0] + boxes[2]) / 2;
        roi_before.y = (boxes[1] + boxes[3]) / 2;

        for (uint32_t i = 0; i < num_classes; i++) {

            box_encoding_center roi_after_centered;
            roi_after_centered.w = exp_map[deltas[2]] * roi_before.w;
            roi_after_centered.h = exp_map[deltas[3]] * roi_before.h;
            roi_after_centered.x = roi_before.x + float_map[deltas[0]] * roi_before.w;
            roi_after_centered.y = roi_before.y + float_map[deltas[1]] * roi_before.h;

            box_encoding_corner roi_after;
            roi_after.x1 = roi_after_centered.x - roi_after_centered.w / 2;
            roi_after.y1 = roi_after_centered.y - roi_after_centered.h / 2;
            roi_after.x2 = roi_after_centered.x + roi_after_centered.w / 2;
            roi_after.y2 = roi_after_centered.y + roi_after_centered.h / 2;

            out_data[0] = min_i32(max_i32(roi_after.x1, 0), MAX_HALFWORD);
            out_data[1] = min_i32(max_i32(roi_after.y1, 0), MAX_HALFWORD);
            out_data[2] = min_i32(max_i32(roi_after.x2, 0), MAX_HALFWORD);
            out_data[3] = min_i32(max_i32(roi_after.y2, 0), MAX_HALFWORD);

            deltas += ROI_LENGTH;
            out_data += ROI_LENGTH;
        }

        boxes += ROI_LENGTH;
    }

    return 0;
}

/**
 * HVX-optimized implementation of the transformation step.
 * @param nn NN graph reference.
 * @param vtd Structure containing op data and parameters.
 */
static void axis_aligned_bbox_transform_execute_hvx(struct nn_graph *nn, void *vtd)
{
    struct tdata *td = vtd;
    
    uint16_t* boxes = td->boxes;
    uint8_t* deltas = td->deltas;
    float deltas_min = td->deltas_min;
    float deltas_max = td->deltas_max;
    uint32_t num_boxes = td->num_boxes;
    uint32_t num_classes = td->num_classes;
    uint16_t* out_data = td->out_data;

    // get exponential value maps
    unsigned char exp_map1[UNIQUE_BYTE_VALUES] __attribute__ ((aligned(BYTE_VECTOR_SIZE)));
    unsigned char exp_map2[UNIQUE_BYTE_VALUES] __attribute__ ((aligned(BYTE_VECTOR_SIZE)));
    get_exp_maps_u16(deltas_min, deltas_max, exp_map1, exp_map2);

    // get floating point value maps
    unsigned char float_map1[UNIQUE_BYTE_VALUES] __attribute__ ((aligned(BYTE_VECTOR_SIZE)));
    unsigned char float_map2[UNIQUE_BYTE_VALUES] __attribute__ ((aligned(BYTE_VECTOR_SIZE)));
    get_float_maps_i16(deltas_min, deltas_max, float_map1, float_map2);

    // There are multiple classes for each box, so box coordinates must be replicated
    // for each class before HVX can use them with the delta tensor. These are buffers
    // for storing replicated data.
    uint16_t roi_before_w_tiled[HALFWORD_VECTOR_SIZE];
    uint16_t roi_before_h_tiled[HALFWORD_VECTOR_SIZE];
    uint16_t roi_before_x_tiled[HALFWORD_VECTOR_SIZE];
    uint16_t roi_before_y_tiled[HALFWORD_VECTOR_SIZE];

    // number of times an individual box was replicated
    int copied = 0;

    // Box data will be replicated with a series of splats and muxes. Here, we create vector predicates for those
    // muxes. Each predicate has 0s at the beginning and 1s in the remaining locations. They are stored in an array
    // in such a way that the array index corresponds to the location of the first bit with a 1.
    HVX_VectorPred border_cases[HALFWORD_VECTOR_SIZE];
    HVX_Vector mask_vector = Q6_V_vsplat_R(0xFFFFFFFF);
    uint16_t* mask_vector_ptr = (uint16_t*)&mask_vector;
    for(int i = 0; i < HALFWORD_VECTOR_SIZE; i++){
        border_cases[i] = Q6_Q_vand_VR(mask_vector, 0xFFFFFFFF);
        *mask_vector_ptr = 0;
        mask_vector_ptr++;
    }

    // vector of zeros for the lower bound of the final result
    HVX_Vector zero_vector = Q6_V_vsplat_R(0);

    // 4 HVXVector's worth of boxes will be processed at a time. These are their offsets.
    const int vector2_offset = HALFWORD_VECTOR_SIZE;
    const int vector3_offset = 2*HALFWORD_VECTOR_SIZE;
    const int vector4_offset = 3*HALFWORD_VECTOR_SIZE;

    // number of chunks to process at a time
    int chunks = num_boxes * ROI_LENGTH / HVX_BOXES_CHUNK_SIZE;

    for(int b = 0; b < chunks; b++){


        // The coordinates are given as 4-tuples consisting of lower-right vertex and upper-right vertex coordinates.
        // ie. Each 4-tuple looks like {x1, y1, x2, y2}. It is not convenient to work with them like that so they
        // will be transposed to 4 HVX vectors: one for x1 values, one for y1, one for x2 and one for y2.

        HVX_Vector *boxes1 = (HVX_Vector *) boxes;
        HVX_Vector *boxes2 = (HVX_Vector *) (boxes + vector2_offset);
        HVX_Vector *boxes3 = (HVX_Vector *) (boxes + vector3_offset);
        HVX_Vector *boxes4 = (HVX_Vector *) (boxes + vector4_offset);

        HVX_VectorPair xy_split1 = Q6_W_vshuff_VVR(*boxes2, *boxes1, 2);
        HVX_VectorPair xy_split2 = Q6_W_vshuff_VVR(*boxes4, *boxes3, 2);

        HVX_Vector xs1 = Q6_V_lo_W(xy_split1);
        HVX_Vector xs2 = Q6_V_lo_W(xy_split2);
        HVX_Vector ys1 = Q6_V_hi_W(xy_split1);
        HVX_Vector ys2 = Q6_V_hi_W(xy_split2);

        HVX_Vector xs1_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(xs1));
        HVX_Vector xs2_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(xs2));
        HVX_Vector ys1_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(ys1));
        HVX_Vector ys2_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(ys2));

        HVX_VectorPair x_split = Q6_W_vdeal_VVR(xs2_sorted, xs1_sorted, 64);
        HVX_VectorPair y_split = Q6_W_vdeal_VVR(ys2_sorted, ys1_sorted, 64);

        HVX_Vector x1 = Q6_V_lo_W(x_split);
        HVX_Vector y1 = Q6_V_lo_W(y_split);
        HVX_Vector x2 = Q6_V_hi_W(x_split);
        HVX_Vector y2 = Q6_V_hi_W(y_split);


		// calculate width, height, x center and y center
        HVX_Vector w = Q6_Vuh_vsub_VuhVuh_sat(x2, x1);
        HVX_Vector h = Q6_Vuh_vsub_VuhVuh_sat(y2, y1);
        HVX_Vector x = Q6_Vuh_vavg_VuhVuh(x2, x1);
        HVX_Vector y = Q6_Vuh_vavg_VuhVuh(y2, y1);

        // these are pointers to the box that is next in line to be replicated
        uint16_t* src_w = (uint16_t*)&w;
        uint16_t* src_h = (uint16_t*)&h;
        uint16_t* src_x = (uint16_t*)&x;
        uint16_t* src_y = (uint16_t*)&y;

        for (uint32_t i = 0; i < num_classes; i++) {

			// pointers to the buffers for replicated boxes
            HVX_Vector* dst_w = (HVX_Vector *) roi_before_w_tiled;
            HVX_Vector* dst_h = (HVX_Vector *) roi_before_h_tiled;
            HVX_Vector* dst_x = (HVX_Vector *) roi_before_x_tiled;
            HVX_Vector* dst_y = (HVX_Vector *) roi_before_y_tiled;

            // first, splat the current box's value to the whole vector
            *dst_w = Q6_V_vsplat_R(*src_w | (*src_w << HALFWORD_BITS));
            *dst_h = Q6_V_vsplat_R(*src_h | (*src_h << HALFWORD_BITS));
            *dst_x = Q6_V_vsplat_R(*src_x | (*src_x << HALFWORD_BITS));
            *dst_y = Q6_V_vsplat_R(*src_y | (*src_y << HALFWORD_BITS));

            // if that splat copied the box values exactly the number of times that was necessary then just
            // move to the next box
            if(copied + HALFWORD_VECTOR_SIZE == num_classes){
                src_w++;
                src_h++;
                src_x++;
                src_y++;
                copied = 0;
            }
            // if the splat copied too many times then splat the next box into a separate vector and use the
            // vector predicates to mux it to the result - repeat this as many times as necessary to fill the
            // buffers
            else if(copied + HALFWORD_VECTOR_SIZE > num_classes){
                int last_idx = num_classes - copied;
                for(int next_box_idx = last_idx; next_box_idx < HALFWORD_VECTOR_SIZE; next_box_idx+=num_classes){
                    src_w++;
                    src_h++;
                    src_x++;
                    src_y++;

                    HVX_Vector next_w = Q6_V_vsplat_R(*src_w | (*src_w << HALFWORD_BITS));
                    HVX_Vector next_h = Q6_V_vsplat_R(*src_h | (*src_h << HALFWORD_BITS));
                    HVX_Vector next_x = Q6_V_vsplat_R(*src_x | (*src_x << HALFWORD_BITS));
                    HVX_Vector next_y = Q6_V_vsplat_R(*src_y | (*src_y << HALFWORD_BITS));

                    *dst_w = Q6_V_vmux_QVV(border_cases[next_box_idx], next_w, *dst_w);
                    *dst_h = Q6_V_vmux_QVV(border_cases[next_box_idx], next_h, *dst_h);
                    *dst_x = Q6_V_vmux_QVV(border_cases[next_box_idx], next_x, *dst_x);
                    *dst_y = Q6_V_vmux_QVV(border_cases[next_box_idx], next_y, *dst_y);

                    copied = min_i32(num_classes, HALFWORD_VECTOR_SIZE - next_box_idx);
                }
                if(copied == num_classes) copied = 0;
            }
            // if the splat did not copy all the necessary values then they will need to be copied and processed
            // on the next iteration - make note of how many were copied and move on
            else{
                copied += HALFWORD_VECTOR_SIZE;
            }

			// Similarly to the box tensor, the deltas are given in 4-tuples of width, height and center x and
            // y deltas. We need to transpose them into one vector for each of those features. In addition, the
            // data is given as quantized 8-bit values so we need to map them to 16-bit values before we can use them.

			HVX_Vector *deltas1 = (HVX_Vector *) deltas;
			HVX_Vector *deltas2 = (HVX_Vector *) (deltas + BYTE_VECTOR_SIZE);

			HVX_VectorPair xy_wh_split = Q6_W_vshuff_VVR(*deltas2, *deltas1, 2);
			HVX_Vector xy = Q6_V_lo_W(xy_wh_split);
			HVX_Vector wh = Q6_V_hi_W(xy_wh_split);

            // get the high and low bytes of the corresponding 16-bit values
			HVX_Vector xy_i16_1;
			HVX_Vector xy_i16_2;
			map_values(&xy, &xy_i16_1, float_map1);
			map_values(&xy, &xy_i16_2, float_map2);

            // shuffle the high and low bytes to get the complete 16-bit values
            HVX_VectorPair xy_i16 = Q6_W_vshuff_VVR(xy_i16_2, xy_i16_1, -1);
            HVX_Vector xy1 = Q6_V_lo_W(xy_i16);
            HVX_Vector xy2 = Q6_V_hi_W(xy_i16);

            // similarly, get the high and low bytes of the correponding 16-bit exponential values
			HVX_Vector wh_u16_1;
			HVX_Vector wh_u16_2;
			map_values(&wh, &wh_u16_1, exp_map1);
			map_values(&wh, &wh_u16_2, exp_map2);

            // shuffle the high and low bytes to get the complete 16-bit values
            HVX_VectorPair wh_u16 = Q6_W_vshuff_VVR(wh_u16_2, wh_u16_1, -1);
            HVX_Vector wh1 = Q6_V_lo_W(wh_u16);
            HVX_Vector wh2 = Q6_V_hi_W(wh_u16);

            // the data is interleaved at this point, these vdeal calls will sort them
			HVX_Vector xy1_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(xy1));
			HVX_Vector xy2_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(xy2));
			HVX_Vector wh1_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(wh1));
			HVX_Vector wh2_sorted = Q6_Vh_vdeal_Vh(Q6_Vh_vdeal_Vh(wh2));

            // data is still interleaved at 32-byte boundaries, this final vdeal will re-arrange them correctly
			HVX_VectorPair xy_split = Q6_W_vdeal_VVR(xy2_sorted, xy1_sorted, 32);
			HVX_VectorPair wh_split = Q6_W_vdeal_VVR(wh2_sorted, wh1_sorted, 32);

            // grab the center x, y and exponential width and height deltas
            HVX_Vector deltas_x = Q6_V_lo_W(xy_split);
            HVX_Vector deltas_y = Q6_V_hi_W(xy_split);
            HVX_Vector deltas_w_exp = Q6_V_lo_W(wh_split);
            HVX_Vector deltas_h_exp = Q6_V_hi_W(wh_split);


            // at this point we have the data in the format we can use it so we start the actual calculation

            HVX_Vector *roi_before_w_tiled_v = (HVX_Vector *) roi_before_w_tiled;
            HVX_Vector *roi_before_h_tiled_v = (HVX_Vector *) roi_before_h_tiled;
            HVX_Vector *roi_before_x_tiled_v = (HVX_Vector *) roi_before_x_tiled;
            HVX_Vector *roi_before_y_tiled_v = (HVX_Vector *) roi_before_y_tiled;

            // w = e^d2 * w;
            // h = e^d3 * h;
            // x = d0 * w;
            // y = d1 * h;
            HVX_VectorPair roi_after_centered_w = Q6_Wuw_vmpy_VuhVuh(deltas_w_exp, *roi_before_w_tiled_v);
            HVX_VectorPair roi_after_centered_h = Q6_Wuw_vmpy_VuhVuh(deltas_h_exp, *roi_before_h_tiled_v);
            HVX_VectorPair roi_after_centered_x = Q6_Ww_vmpy_VhVuh(deltas_x, *roi_before_w_tiled_v);
            HVX_VectorPair roi_after_centered_y = Q6_Ww_vmpy_VhVuh(deltas_y, *roi_before_h_tiled_v);

            // when deltas were mapped to 16-bit values, they were multiplied by a constant to preserve accuracy
            // here, we divide (ie. right-shift) by that constant to get the correct value
            HVX_Vector roi_after_centered_w_1 = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(roi_after_centered_w), EXP_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_w_2 = Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(roi_after_centered_w), EXP_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_h_1 = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(roi_after_centered_h), EXP_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_h_2 = Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(roi_after_centered_h), EXP_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_x_1 = Q6_Vw_vasr_VwR(Q6_V_lo_W(roi_after_centered_x), FLT_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_x_2 = Q6_Vw_vasr_VwR(Q6_V_hi_W(roi_after_centered_x), FLT_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_y_1 = Q6_Vw_vasr_VwR(Q6_V_lo_W(roi_after_centered_y), FLT_ACCURACY_SHIFT);
            HVX_Vector roi_after_centered_y_2 = Q6_Vw_vasr_VwR(Q6_V_hi_W(roi_after_centered_y), FLT_ACCURACY_SHIFT);

            // shuffle to get the right ordering
            roi_after_centered_w = Q6_W_vshuff_VVR(roi_after_centered_w_2, roi_after_centered_w_1, -4);
            roi_after_centered_h = Q6_W_vshuff_VVR(roi_after_centered_h_2, roi_after_centered_h_1, -4);
            roi_after_centered_x = Q6_W_vshuff_VVR(roi_after_centered_x_2, roi_after_centered_x_1, -4);
            roi_after_centered_y = Q6_W_vshuff_VVR(roi_after_centered_y_2, roi_after_centered_y_1, -4);

            // grab the ordered values
            roi_after_centered_w_1 = Q6_V_lo_W(roi_after_centered_w);
            roi_after_centered_w_2 = Q6_V_hi_W(roi_after_centered_w);
            roi_after_centered_h_1 = Q6_V_lo_W(roi_after_centered_h);
            roi_after_centered_h_2 = Q6_V_hi_W(roi_after_centered_h);
            roi_after_centered_x_1 = Q6_V_lo_W(roi_after_centered_x);
            roi_after_centered_x_2 = Q6_V_hi_W(roi_after_centered_x);
            roi_after_centered_y_1 = Q6_V_lo_W(roi_after_centered_y);
            roi_after_centered_y_2 = Q6_V_hi_W(roi_after_centered_y);

            // the above multiplication resulted in 32-bit values so we have to expand the operands for the remaining math
            HVX_VectorPair roi_before_x_tiled_i32 = Q6_Ww_vunpack_Vh(*roi_before_x_tiled_v);
            HVX_VectorPair roi_before_y_tiled_i32 = Q6_Ww_vunpack_Vh(*roi_before_y_tiled_v);

            // x = x_original + x;
            // y = y_original + y;
            roi_after_centered_x_1 = Q6_Vw_vadd_VwVw(roi_after_centered_x_1, Q6_V_lo_W(roi_before_x_tiled_i32));
            roi_after_centered_x_2 = Q6_Vw_vadd_VwVw(roi_after_centered_x_2, Q6_V_hi_W(roi_before_x_tiled_i32));
            roi_after_centered_y_1 = Q6_Vw_vadd_VwVw(roi_after_centered_y_1, Q6_V_lo_W(roi_before_y_tiled_i32));
            roi_after_centered_y_2 = Q6_Vw_vadd_VwVw(roi_after_centered_y_2, Q6_V_hi_W(roi_before_y_tiled_i32));

            // we need to add/subtract half of width and height so we divide them by 2 first
            roi_after_centered_w_1 = Q6_Vuw_vlsr_VuwR(roi_after_centered_w_1, 1);
            roi_after_centered_w_2 = Q6_Vuw_vlsr_VuwR(roi_after_centered_w_2, 1);
            roi_after_centered_h_1 = Q6_Vuw_vlsr_VuwR(roi_after_centered_h_1, 1);
            roi_after_centered_h_2 = Q6_Vuw_vlsr_VuwR(roi_after_centered_h_2, 1);

            // x1 = x - w / 2;
            // y1 = y - h / 2;
            // x2 = x + w / 2;
            // y2 = y + h / 2;
            HVX_Vector roi_after_x1_1 = Q6_Vw_vsub_VwVw(roi_after_centered_x_1, roi_after_centered_w_1);
            HVX_Vector roi_after_x1_2 = Q6_Vw_vsub_VwVw(roi_after_centered_x_2, roi_after_centered_w_2);
            HVX_Vector roi_after_y1_1 = Q6_Vw_vsub_VwVw(roi_after_centered_y_1, roi_after_centered_h_1);
            HVX_Vector roi_after_y1_2 = Q6_Vw_vsub_VwVw(roi_after_centered_y_2, roi_after_centered_h_2);
            HVX_Vector roi_after_x2_1 = Q6_Vw_vadd_VwVw(roi_after_centered_x_1, roi_after_centered_w_1);
            HVX_Vector roi_after_x2_2 = Q6_Vw_vadd_VwVw(roi_after_centered_x_2, roi_after_centered_w_2);
            HVX_Vector roi_after_y2_1 = Q6_Vw_vadd_VwVw(roi_after_centered_y_1, roi_after_centered_h_1);
            HVX_Vector roi_after_y2_2 = Q6_Vw_vadd_VwVw(roi_after_centered_y_2, roi_after_centered_h_2);

            // make sure we're not going lower than 0
            roi_after_x1_1 = Q6_Vw_vmax_VwVw(roi_after_x1_1, zero_vector);
            roi_after_x1_2 = Q6_Vw_vmax_VwVw(roi_after_x1_2, zero_vector);
            roi_after_y1_1 = Q6_Vw_vmax_VwVw(roi_after_y1_1, zero_vector);
            roi_after_y1_2 = Q6_Vw_vmax_VwVw(roi_after_y1_2, zero_vector);
            roi_after_x2_1 = Q6_Vw_vmax_VwVw(roi_after_x2_1, zero_vector);
            roi_after_x2_2 = Q6_Vw_vmax_VwVw(roi_after_x2_2, zero_vector);
            roi_after_y2_1 = Q6_Vw_vmax_VwVw(roi_after_y2_1, zero_vector);
            roi_after_y2_2 = Q6_Vw_vmax_VwVw(roi_after_y2_2, zero_vector);

            // convert the 32-bit values to 16-bit
            HVX_Vector roi_after_x1 = Q6_Vuh_vpack_VwVw_sat(roi_after_x1_2, roi_after_x1_1);
            HVX_Vector roi_after_y1 = Q6_Vuh_vpack_VwVw_sat(roi_after_y1_2, roi_after_y1_1);
            HVX_Vector roi_after_x2 = Q6_Vuh_vpack_VwVw_sat(roi_after_x2_2, roi_after_x2_1);
            HVX_Vector roi_after_y2 = Q6_Vuh_vpack_VwVw_sat(roi_after_y2_2, roi_after_y2_1);


            // format the data back to 4-tuple format
            HVX_VectorPair roi_after_x1y1 = Q6_W_vshuff_VVR(roi_after_y1, roi_after_x1, -2);
            HVX_VectorPair roi_after_x2y2 = Q6_W_vshuff_VVR(roi_after_y2, roi_after_x2, -2);

            HVX_VectorPair roi_after_final1 = Q6_W_vshuff_VVR(Q6_V_lo_W(roi_after_x2y2), Q6_V_lo_W(roi_after_x1y1), -4);
            HVX_VectorPair roi_after_final2 = Q6_W_vshuff_VVR(Q6_V_hi_W(roi_after_x2y2), Q6_V_hi_W(roi_after_x1y1), -4);


            // store the final output
            HVX_Vector *vout1 = (HVX_Vector*)out_data;
            HVX_Vector *vout2 = (HVX_Vector*)(out_data+vector2_offset);
            HVX_Vector *vout3 = (HVX_Vector*)(out_data+vector3_offset);
            HVX_Vector *vout4 = (HVX_Vector*)(out_data+vector4_offset);

            *vout1 = Q6_V_lo_W(roi_after_final1);
            *vout2 = Q6_V_hi_W(roi_after_final1);
            *vout3 = Q6_V_lo_W(roi_after_final2);
            *vout4 = Q6_V_hi_W(roi_after_final2);

            out_data += HVX_BOXES_CHUNK_SIZE;
			deltas += HVX_BOXES_CHUNK_SIZE;
        }

        boxes += HVX_BOXES_CHUNK_SIZE;
    }

    // if the number of boxes is not a perfect multiple of 4 HVX vector's worth of data then process the remainder
    // with the reference implementation
    if(chunks * HVX_BOXES_CHUNK_SIZE != num_boxes * ROI_LENGTH){
        uint16_t* boxes_data_end = td->boxes + num_boxes * ROI_LENGTH;

        axis_aligned_bbox_transform_execute_ref(nn, boxes, deltas_min, deltas_max, num_classes, boxes_data_end, deltas, out_data);
    }

    nn_sem_post(&td->donesem);
}

uint16_t requantize(uint16_t value, float min, float max){
    float step = (max - min) / 65536.f;
    float float_val = min + value * step;
    return (uint16_t)(float_val / BOX_DIM_STEP);
}

static int axis_aligned_bbox_transform_execute(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"axis_aligned_bbox_transform execute. self=%p ",self);

    const struct tensor *boxes_input_tensor = self->inputs[0];
    const struct tensor *deltas_input_tensor = self->inputs[1];
    const struct tensor *batch_splits_input_tensor = self->inputs[2];
    const struct tensor *image_info_input_tensor = self->inputs[3];

    struct tensor *boxes_output_tensor = self->outputs[0];

    uint16_t *boxes_input = boxes_input_tensor->data;
    uint8_t *deltas_input = deltas_input_tensor->data;
    int *batch_splits = batch_splits_input_tensor->data;
    uint16_t *image_info = image_info_input_tensor->data;
    float deltas_min = tensor_get_float(self->inputs[4], 0);
    float deltas_max = tensor_get_float(self->inputs[5], 0);
    float image_info_min = tensor_get_float(self->inputs[6], 0);
    float image_info_max = tensor_get_float(self->inputs[7], 0);

    uint16_t *boxes_output = boxes_output_tensor->data;

    tensor_out_prepare_normal(boxes_output_tensor,    1,1,deltas_input_tensor->shape.width,deltas_input_tensor->shape.depth, NN_TYPE_QUINT16);

    uint32_t num_boxes = boxes_input_tensor->shape.width;
    uint32_t num_classes = deltas_input_tensor->shape.depth / ROI_LENGTH;
    uint32_t num_batches = image_info_input_tensor->shape.width;

    uint16_t* boxes = boxes_input;
    uint16_t* boxes_data_end = boxes_input + num_boxes * ROI_LENGTH;
    uint8_t* deltas = deltas_input;
    uint16_t* out_data = boxes_output;
    uint32_t roiIndex = 0;

    // if there are less than 4 HVX vector's worth of box data then there is not enough of it to use HVX
    if(num_boxes < HVX_BOXES_CHUNK_SIZE){
        int result = axis_aligned_bbox_transform_execute_ref(nn, boxes, deltas_min, deltas_max, num_classes, boxes_data_end, deltas, out_data);

        if(result) return result;
    }
    else{
        struct tdata td;
        td.boxes = boxes;
        td.deltas = deltas;
        td.deltas_min = deltas_min;
        td.deltas_max = deltas_max;
        td.num_boxes = num_boxes;
        td.num_classes = num_classes;
        td.out_data = out_data;

        nn_sem_init(&td.donesem, 0);
        nn_os_work_for_vector(nn, axis_aligned_bbox_transform_execute_hvx, &td);
        nn_sem_wait(&td.donesem);
    }

    // limit the final results to the max box sizes
    const uint32_t imageLength = 2;
    out_data = boxes_output;

    for (boxes = boxes_input; boxes < boxes_data_end; boxes += ROI_LENGTH, roiIndex++) {

        uint32_t batchIndex = batch_splits[roiIndex];

        if (batchIndex >= num_batches) return errlog(nn,"batchIndex is not less than num_batches");

        const uint16_t* imageInfoBase = image_info + batchIndex * imageLength;
        uint16_t imageHeight = imageInfoBase[0];
        uint16_t imageWidth = imageInfoBase[1];

        // only re-quantize if the image_info quantization was different from the boxes
        if(image_info_min != 0.f || image_info_max != 8192.f) {
            imageHeight = requantize(imageInfoBase[0], image_info_min, image_info_max);
            imageWidth = requantize(imageInfoBase[1], image_info_min, image_info_max);
        }

        for (uint32_t i = 0; i < num_classes; i++) {

            out_data[0] = min_i32(out_data[0], imageWidth);
            out_data[1] = min_i32(out_data[1], imageHeight);
            out_data[2] = min_i32(out_data[2], imageWidth);
            out_data[3] = min_i32(out_data[3], imageHeight);

            out_data += ROI_LENGTH;
        }
    }

    return 0;
}


struct nn_node_ops nn_ops_for_AxisAlignedBBoxTransform_q8q16 = {
        .execute = axis_aligned_bbox_transform_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(8),
        .n_outputs = NN_IOCOUNT(1),
};

