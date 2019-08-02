/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
 * This contains implementations for Resize, using unit square algorithm
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>

static size_t min_sizet(size_t a, size_t b) { return((a<b) ? a : b); }

// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_RESIZE_INPUT_DATA_IDX 0
#define OP_RESIZE_SCALEW_IDX 1
#define OP_RESIZE_SCALEH_IDX 2
#define OP_RESIZE_PADVALUE_IDX 3
#define OP_RESIZE_MAINTAINASPECT_IDX 4
#define OP_RESIZE_OUTPUT_SHAPE_IDX 5
#define OP_RESIZE_NUM_OPS 6

 static int resize_unitsquare_execute(struct nn_node *self, struct nn_graph *nn)
 {

 	const struct tensor *in_tensor = self->inputs[OP_RESIZE_INPUT_DATA_IDX];
 	const struct tensor *scalew_tensor = self->inputs[OP_RESIZE_SCALEW_IDX];
 	const struct tensor *scaleh_tensor = self->inputs[OP_RESIZE_SCALEH_IDX];
 	const struct tensor *padvalue_tensor = self->inputs[OP_RESIZE_PADVALUE_IDX];
 	const struct tensor *maintainaspect_tensor = self->inputs[OP_RESIZE_MAINTAINASPECT_IDX];
 	const struct tensor *output_shape_tensor = self->inputs[OP_RESIZE_OUTPUT_SHAPE_IDX];


 	struct tensor* out_tensor = self->outputs[0];
 	float *input = (float*) in_tensor->data;

 	tensor_set_shape(out_tensor,
 		output_shape_tensor->shape.batches, output_shape_tensor->shape.height,
 		output_shape_tensor->shape.width, output_shape_tensor->shape.depth);
 	out_tensor->data_size = output_shape_tensor->shape.height *
 	output_shape_tensor->shape.width *
 	output_shape_tensor->shape.depth * sizeof(float);

 	float scaleW = tensor_get_float(scalew_tensor, 0);
 	float scaleH = tensor_get_float(scaleh_tensor, 0);
 	float padvalue = tensor_get_float(padvalue_tensor, 0);
 	uint32_t maintainAspectRatio = tensor_get_int32(maintainaspect_tensor, 0);
    // using unit square formula: f(x,y) = f00 + f10*x + f01*y + f11*x*y
    // where f00 = f(0,0)
    // f10 = f(1,0) - f(0,0)
    // f01 = f(0,1) - f(0,0)
    // f11 = f(1,1) + f(0,0) - (f(1,0) + f(0,1))
 	const size_t depth = in_tensor->shape.depth;
 	const size_t inputWidth = in_tensor->shape.width;
 	const size_t inputHeight = in_tensor->shape.height;
 	const size_t outputWidth = output_shape_tensor->shape.width;
 	const size_t outputHeight = output_shape_tensor->shape.height;
 	const size_t targetWidth = inputWidth * scaleW;
 	const size_t targetHeight = inputHeight * scaleH;
 	const size_t offsetY = (outputHeight - targetHeight) /2;
 	const size_t offsetX = (outputWidth - targetWidth) / 2;

 	for(size_t y = 0; y < outputHeight; ++y ) {
 		int padH = maintainAspectRatio &&
 		(((float)y < (outputHeight - targetHeight) *0.5f) ||
 		 ((float)y > (outputHeight + targetHeight) *0.5f));

 		const float srcY = (float)(y-offsetY) / scaleH;
 		const size_t y0 = floorf(srcY);
 		const size_t y1 = min_sizet((size_t)ceilf(srcY),inputHeight-1);
 		const float py = srcY - y0;

 		for(size_t x = 0; x < outputWidth; ++x )
 		{
 			int padW = maintainAspectRatio &&
 			(((float)x < (outputWidth - targetWidth) *0.5f) ||
 			 ((float)x > (outputWidth + targetWidth) *0.5f));

 			if (padH || padW)
 			{
 				float* output = (float*)out_tensor->data + depth*(x + outputWidth * y);
 				for(size_t z = 0; z < depth; ++z) {
 					output[z] = padvalue;
 				}
 			} else {
 				const float srcX = (float)(x-offsetX) / scaleW;
 				const size_t x0 = floorf(srcX);
 				const size_t x1 = min_sizet((size_t)ceil(srcX),inputWidth-1);
 				const float px = srcX - x0;

 				const float* f00 = input + depth*(x0 + inputWidth*y0);
 				const float* f01 = input + depth*(x0 + inputWidth*y1);
 				const float* f10 = input + depth*(x1 + inputWidth*y0);
 				const float* f11 = input + depth*(x1 + inputWidth*y1);
 				float* output = (float*) out_tensor->data + depth*(x + outputWidth*y);

 				for(size_t z = 0; z < depth; ++z)
 				{
 					const float a00 = f00[z];
 					const float a01 = f01[z] - a00;
 					const float a10 = f10[z] - a00;
 					const float a11 = f11[z] - a10 - a01 - a00;
 					output[z] = a00 + a10*px + a01*py + a11*px*py;

 				}
 			}
 		}
    }
	return 0;
}


struct nn_node_ops nn_ops_for_ResizeUnitSquare_f = {
	.execute = resize_unitsquare_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(OP_RESIZE_NUM_OPS),
	.n_outputs = NN_IOCOUNT(1),

};


