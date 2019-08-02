
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
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains implementations for quantized sqrt node
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

struct tdata {
    const uint8_t *in_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

static void sqrt_execute_hvx(struct nn_graph *nn, void *vtd)
{
    static const unsigned char sqrt_lut[256] __attribute__ ((aligned(128))) = {
            0, 16, 23, 28, 32, 36, 39, 42, 45, 48, 50, 53, 55, 58, 60, 62, 64, 66, 68, 70, 71, 73, 75, 77, 78, 80, 81,
            83, 84, 86, 87, 89, 90, 92, 93, 94, 96, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113,
            114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
            135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 145, 146, 147, 148, 149, 150, 151, 151, 152, 153,
            154, 155, 156, 156, 157, 158, 159, 160, 160, 161, 162, 163, 164, 164, 165, 166, 167, 167, 168, 169, 170,
            170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 179, 179, 180, 181, 181, 182, 183, 183, 184, 185,
            186, 186, 187, 188, 188, 189, 190, 190, 191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199,
            199, 200, 201, 201, 202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211, 212,
            212, 213, 214, 214, 215, 215, 216, 217, 217, 218, 218, 219, 220, 220, 221, 221, 222, 222, 223, 224, 224,
            225, 225, 226, 226, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232, 233, 233, 234, 234, 235, 235, 236,
            236, 237, 237, 238, 238, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247,
            247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255
    };

    struct tdata *td = vtd;
    uint8_t* in_data = (uint8_t*) td->in_data;
    uint8_t* out_data = td->out_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128); //ceiling
    // byte shuffle table
    HVX_Vector luta = *(HVX_Vector *) sqrt_lut;
    HVX_Vector lutb = *(HVX_Vector *) & sqrt_lut[128];
    HVX_Vector lut0 = Q6_Vb_vshuff_Vb(luta);
    HVX_Vector lut1 = Q6_Vb_vshuff_Vb(lutb);

    for (int i=0; i<num_loops; i++) {
        HVX_Vector *vin = (HVX_Vector *) in_data;
        HVX_Vector *vout = (HVX_Vector *) out_data;
        /*
         * look up value in table
         * only 32 bytes can be done at a time, so we need to do 8 lookups and OR the results
         */
        HVX_Vector v_in = *vin;
        HVX_Vector result = q6op_Vb_vlut32_VbVbI(v_in, lut0, 0);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut0, 1);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut0, 2);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut0, 3);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut1, 4);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut1, 5);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut1, 6);
        result = q6op_Vb_vlut32or_VbVbVbI(result, v_in, lut1, 7);
        *vout = result;
        // move pointers to process next 128 bytes
        in_data += 128;
        out_data += 128;
    }
    nn_sem_post(&td->donesem);
}

static int sqrt_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    if (tensor_get_float(in_min_tensor,0) < 0) {
        return errlog(nn, "Current sqrt implementation does not support negative numbers");
    }

    const struct shape in_shape = in_tensor->shape;

    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    struct tdata td = {
            .in_data = in_data,
            .out_data = out_data,
            .num_elements = in_shape.batches * in_shape.height * in_shape.width * in_shape.depth
    };
    nn_sem_init(&td.donesem,0);

    if (tensor_out_prepare_normal_fromshape( out_tensor, & in_tensor->shape, NN_TYPE_QUINT8 )!= 0) {
        return errlog(nn,"out too small");
    }
    if (tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) {
        return errlog(nn,"out min too small");
    }
    if (tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) {
        return errlog(nn,"out max too small");
    }

    nn_os_work_for_vector(nn,sqrt_execute_hvx,&td);
    nn_sem_wait(&td.donesem);

    // set the output min and max to the sqrt of the input min and max
    tensor_set_float(out_min_tensor,0,sqrtf(tensor_get_float(in_min_tensor,0)));
    tensor_set_float(out_max_tensor,0,sqrtf(tensor_get_float(in_max_tensor,0)));

    return 0;
}


struct nn_node_ops nn_ops_for_QuantizedSqrt_8 = {
        .execute = sqrt_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(3),
};
