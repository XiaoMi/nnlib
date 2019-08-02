
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
#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains an extract_glimpse node
 */

static float get_pseudo_gaussian_random(float mean, float standard_deviation){
	float result = 0.f;
	long int sum = 0;

	for(int i = 0; i < 12; i++) sum += rand();
	
	result = (float)sum/(float)RAND_MAX;
	
	result = result - 6.f;
	
	result = standard_deviation * result + mean;
	
	if(result > 255.f) result = 255.f;
	if(result < 0.f) result = 0.f;
	
	return result;
}

static int extract_glimpse_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *image_tensor = self->inputs[0];
    const struct tensor *offsets_tensor = self->inputs[1];
    
    struct tensor *output_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint8_t* image = image_tensor->data;
	float in_min = tensor_get_float(self->inputs[2], 0);
	float in_max = tensor_get_float(self->inputs[3], 0);
    float* offsets = offsets_tensor->data;
    int glimpse_width = tensor_get_int32(self->inputs[4], 0);
    int glimpse_height = tensor_get_int32(self->inputs[5], 0);
    int centered = tensor_get_int32(self->inputs[6], 0);
    int normalized = tensor_get_int32(self->inputs[7], 0);
    int uniform_noise = tensor_get_int32(self->inputs[8], 0);

	uint8_t* output = output_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	int batch_size = image_tensor->shape.batches;
	int image_height = image_tensor->shape.height;
	int image_width = image_tensor->shape.width;
	int image_depth = image_tensor->shape.depth;
	int image_size = image_width * image_height * image_depth;
	int glimpse_size = glimpse_width * glimpse_height * image_depth;

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(output_tensor, batch_size, glimpse_height, glimpse_width, image_depth, NN_TYPE_QUINT8);
	
	*out_min = in_min;
	*out_max = in_max;

    for (int i = 0; i < batch_size; i++) {
		uint8_t* outputBegin = output + (i * glimpse_size);
        float offset_x = offsets[2*i+1];
        float offset_y = offsets[2*i];
        int current_glimpse_width = glimpse_width;
        int current_glimpse_height = glimpse_height;
        int padding_required = 0;

        if (normalized) {
            offset_x = offset_x * image_width;
            offset_y = offset_y * image_height;
        }

        if (centered) {
            offset_x = (offset_x + image_width) / 2.f;
            offset_y = (offset_y + image_height) / 2.f;
        }

        // Offset coordinates are at middle of the glimpse window so need to shift offsets to get
        // to the start of the glimpse window
        offset_x = offset_x - ((float)current_glimpse_width / 2.f) + 0.5f;
        offset_y = offset_y - ((float)current_glimpse_height / 2.f) + 0.5f;

        // Offsets should be integers now so convert it
        int offset_x_input = (int)offset_x;
        int offset_y_input = (int)offset_y;
        int offset_x_output = 0;
        int offset_y_output = 0;

        if (offset_x_input < 0) {
            padding_required = 1;
            current_glimpse_width = glimpse_width + offset_x_input > 0 ? glimpse_width + offset_x_input : 0;
            offset_x_input = 0;
            offset_x_output = glimpse_width - current_glimpse_width;
        }
        else if (offset_x_input + current_glimpse_width > image_width) {
            padding_required = 1;
            current_glimpse_width = image_width - offset_x_input > 0 ? image_width - offset_x_input : 0;
        }

        if (offset_y_input < 0) {
            padding_required = 1;
            current_glimpse_height = glimpse_height + offset_y_input > 0 ? glimpse_height + offset_y_input : 0;
            offset_y_input = 0;
            offset_y_output = glimpse_height - current_glimpse_height;
        }
        else if (offset_y_input + current_glimpse_height > image_height) {
            padding_required = 1;
            current_glimpse_height = image_height - offset_y_input > 0 ? image_height - offset_y_input : 0;
        }

        if (padding_required) {
			srand(0);
            int input_begin = i * image_size;
            int input_end = (i + 1) * image_size;
            if (uniform_noise) {
				uint8_t min = 255;
				uint8_t max = 0;
                for(int j = input_begin; j < input_end; j++){
					if(min > image[j]) min = image[j];
					if(max < image[j]) max = image[j];
				}
				uint8_t range = max - min;
                for(int j = 0; j < glimpse_size; j++){
					outputBegin[j] = rand() % range + min;
				}
            }
            else {
				float mean = 0.f;
				for(int j = input_begin; j < input_end; j++) mean += image[j];
				mean /= (float)(input_end - input_begin);
				
				float variance = 0;
				for(int j = input_begin; j < input_end; j++){
					float difference = image[j] - mean;
					variance += difference * difference;
				}
				variance /= (input_end - input_begin);
				
				float standard_deviation = sqrtf(variance);
				
				for(int j = 0; j < glimpse_size; j++){
					outputBegin[j] = round(get_pseudo_gaussian_random(mean, standard_deviation));
				}
            }
        }

        if (current_glimpse_width == 0 || current_glimpse_height == 0){
            continue;
		}

        // Perform the actual slice
		uint8_t* inputBegin = image + (i * image_size) +
                                  image_depth * (offset_x_input + image_width * offset_y_input);
		outputBegin += image_depth * (offset_x_output + glimpse_width * offset_y_output);
        for (int y = 0; y < current_glimpse_height; y++) {
            memcpy(outputBegin, inputBegin, sizeof(uint8_t)*(current_glimpse_width * image_depth));
            inputBegin += (image_width * image_depth);
            outputBegin += (glimpse_width * image_depth);
        }
    }
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedExtractGlimpse_8 = {
	.execute = extract_glimpse_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(9),
	.n_outputs = NN_IOCOUNT(3),
};

