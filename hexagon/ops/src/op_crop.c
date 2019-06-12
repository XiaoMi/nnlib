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
 * This contains implementations for quantized crop node
 */

#include <nn_graph.h>
#include <nn_asm_ops.h>
#include <string.h>

// TODO: Get these into some shareable place.
// Currently we have a copy here and also one in SNPE DSP code
#define OP_CROP_INPUT_DATA_IDX 0
#define OP_CROP_INPUT_MIN_IDX 1
#define OP_CROP_INPUT_MAX_IDX 2
#define OP_CROP_OFFSETS_IDX 3
#define OP_CROP_INPUT_STRIDES_IDX 4
#define OP_CROP_OUTPUT_STRIDES_IDX 5
#define OP_CROP_OUTPUT_SHAPE_IDX 6
#define NUM_CROP_OPS 7
#define BATCH_OFFSET_IDX 0
#define HEIGHT_OFFSET_IDX 1
#define WIDTH_OFFSET_IDX 2
#define DEPTH_OFFSET_IDX 3

#define NUM_INPUT_DIMS 4
 typedef struct CropParms_t {
	int numOffsets;
	int offsets[NUM_INPUT_DIMS];
	int numInStrides;
	int inStrides[NUM_INPUT_DIMS];
	int numOutStrides;
	int outStrides[NUM_INPUT_DIMS];
 } CropParms_t;

static void Crop2D(const struct tensor* input, struct tensor* output, const CropParms_t parms)
{
	uint32_t inputImageStride = input->shape.height * input->shape.width * input->shape.depth;
	uint32_t inputRowStride = input->shape.width * input->shape.depth;
	uint32_t outputRowStride = output->shape.width * output->shape.depth;
	uint32_t inputColOffset = parms.offsets[WIDTH_OFFSET_IDX] * input->shape.depth;
	uint8_t* inputPtr ;
	uint8_t* outputPtr = (uint8_t*) output->data;

	for (int b = 0; b < output->shape.batches; b++ ){
		inputPtr = (uint8_t*) input->data
		+ (inputImageStride * b)
		+ (inputRowStride * parms.offsets[HEIGHT_OFFSET_IDX])
		+ inputColOffset;
		for(int i = 0; i < output->shape.height; i++){
			vmemcpy_asm(outputPtr, inputPtr, outputRowStride);
			inputPtr += inputRowStride;
			outputPtr += outputRowStride;
		}
	}
}

 static void CropOuter(const struct tensor* input,
 	const int inputOffset,
 	struct tensor* output,
 	const int outputOffset,
 	const int axis,
 	const CropParms_t parms)
 {

 	if (axis == (parms.numOffsets - 1)) {
		// actually crop inner
		vmemcpy_asm(((uint8_t*) output->data) + outputOffset,
 			((uint8_t*) input->data) + parms.offsets[axis] + inputOffset,
			*((uint32_t*)(&output->shape.batches) + axis));
 	} else {
		for( size_t i = 0; i < *((uint32_t*)(&output->shape.batches) + axis) ; ++i ) {
			CropOuter(input,
 				inputOffset + (i+parms.offsets[axis])*parms.inStrides[axis],
 				output,
 				outputOffset + i*parms.outStrides[axis],
 				axis+1,
 				parms);
 		}
 	}
 }
//Check to see if h x w crop
static int crop_check_2D(const struct tensor* input, const struct tensor* output, CropParms_t parms) {
	return (input->shape.batches == output->shape.batches && parms.offsets[BATCH_OFFSET_IDX] == 0)
		&& (input->shape.depth == output->shape.depth && parms.offsets[DEPTH_OFFSET_IDX] == 0);
}

 static void crop_execute_worker(struct nn_graph *nn, void *vself)
 {
 	struct nn_node *self = vself;
 	const struct tensor *in_tensor = self->inputs[OP_CROP_INPUT_DATA_IDX];
 	const struct tensor *offsets_tensor = self->inputs[OP_CROP_OFFSETS_IDX];
 	const struct tensor *input_strides_tensor = self->inputs[OP_CROP_INPUT_STRIDES_IDX];
 	const struct tensor *output_strides_tensor = self->inputs[OP_CROP_OUTPUT_STRIDES_IDX];
 	const struct tensor *output_shape_tensor = self->inputs[OP_CROP_OUTPUT_SHAPE_IDX];

 	struct tensor* out_tensor = self->outputs[0];

 	tensor_set_shape(out_tensor,
 		output_shape_tensor->shape.batches, output_shape_tensor->shape.height,
 		output_shape_tensor->shape.width, output_shape_tensor->shape.depth);
	out_tensor->data_size = output_shape_tensor->shape.batches *
	output_shape_tensor->shape.height *
 	output_shape_tensor->shape.width *
 	output_shape_tensor->shape.depth;

	// Set output min and max same as input..
 	tensor_copy(self->outputs[1],self->inputs[OP_CROP_INPUT_MIN_IDX]);
 	tensor_copy(self->outputs[2],self->inputs[OP_CROP_INPUT_MAX_IDX]);

 	CropParms_t parms;

	// Load up constants
	// Offsets
 	parms.numOffsets = offsets_tensor->shape.depth;
 	for (int i= 0; i < parms.numOffsets && i < NUM_INPUT_DIMS; i++) {
 		parms.offsets[i] = tensor_get_int32(offsets_tensor,i);
 	}

	// inputStrides
 	parms.numInStrides = input_strides_tensor->shape.depth;
 	for (int i= 0; i < parms.numInStrides && i < NUM_INPUT_DIMS; i++) {
 		parms.inStrides[i] = tensor_get_int32(input_strides_tensor,i);
 	}

	// outputStrides
 	parms.numOutStrides = output_strides_tensor->shape.depth;
 	for (int i= 0; i < parms.numOutStrides && i < NUM_INPUT_DIMS; i++) {
 		parms.outStrides[i] = tensor_get_int32(output_strides_tensor,i);
 	}
    if(crop_check_2D(in_tensor, out_tensor, parms)) {
		Crop2D(in_tensor, out_tensor, parms);
	}
    else {
		CropOuter(in_tensor, 	// Input data
			0,			// Starting input offset of 0
			out_tensor,	// Output data
			0,			// starting output offset of 0
			0,			// start with offset 0
			parms);
    }
	nn_sem_post(self->opaque);
}

static int crop_execute(struct nn_node *self, struct nn_graph *nn) {
    nn_sem_t donesem;
    nn_sem_init(&donesem,0);
    self->opaque = &donesem;
    nn_os_work_for_vector(nn,crop_execute_worker,self);
    nn_sem_wait(&donesem);
    self->opaque = NULL;
    return 0;
}


struct nn_node_ops nn_ops_for_QuantizedCrop_8 = {
	.execute = crop_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(NUM_CROP_OPS),
	.n_outputs = NN_IOCOUNT(3),
};


