/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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
#include "float_mathops.h"

static inline const float* transform_get_valid_output_location(int32_t x, int32_t x_end, int32_t y, int32_t y_end, 
								const float* start_ptr, int32_t width_stride, int32_t depth_stride){
	if ((x>=0 && x<x_end) && (y>=0 && y<y_end) ){
		int idx = depth_stride * (x + (y * width_stride));
		return start_ptr + idx ;
	}
	return NULL;
}
struct bilinear_interpolation_info{
	const float *top_right_ptr, *top_left_ptr, *bottom_right_ptr, *bottom_left_ptr;
	float y_frac,x_frac;
	float* out_ptr;
};
struct tdata_precalculation{
	// x_out_start;
	int y_out_start;
	int y_out_end;
	int out_width;
	int out_depth;
	int out_height;
	const float* in_ptr;
	float* const out_ptr_start;
	struct bilinear_interpolation_info *blerp_info_list_ptr;
	int32_t *num_interpolation_info;
	float a0;
	float a1;
	float a2;
	float b0;
	float b1;
	float b2;
	float c0;
	float c1;
	nn_sem_t* donesem;
};

static void transform_precalculate_data_location(struct nn_graph* nn, void* args){
	struct tdata_precalculation * tdata = (struct tdata_precalculation*) args;
	int y_out_start = tdata-> y_out_start;
	int out_width = tdata->out_width;
	int out_depth = tdata->out_depth;
	int out_height = tdata->out_height;
	int y_out_end = tdata->y_out_end;
	const float * in_ptr = tdata->in_ptr;
	float * const out_ptr_start = tdata->out_ptr_start;
	struct bilinear_interpolation_info * blerp_info_list_ptr = tdata->blerp_info_list_ptr;
	const float *top_right_ptr, *top_left_ptr, *bottom_right_ptr, *bottom_left_ptr;
	int32_t* num_interpolation_info = tdata->num_interpolation_info;

	*num_interpolation_info = 0;
	float a0 = tdata->a0;
	float a1 = tdata->a1;
	float a2 = tdata->a2;
	float b0 = tdata->b0;
	float b1 = tdata->b1;
	float b2 = tdata->b2;
	float c0 = tdata->c0;
	float c1 = tdata->c1;
	float k;
	float x_in_prime, y_in_prime;
	int32_t x_in_0,x_in_1, y_in_0, y_in_1;
	for (int y_out=y_out_start; y_out< y_out_end; y_out++){ 
		
		for (int x_out=0; x_out< out_width; x_out++){

			k = c0 * x_out + c1 * y_out + 1.f;
			if (k == 0.0f){
				continue;
			}
			x_in_prime = (a0 * x_out + a1 * y_out + a2) / k;
			y_in_prime = (b0 * x_out + b1 * y_out + b2) / k;
			x_in_0 = floorf(x_in_prime);
			x_in_1 = x_in_0 + 1;
			y_in_0 = floorf(y_in_prime);
			y_in_1 = y_in_0 + 1;
			top_right_ptr = transform_get_valid_output_location(x_in_0,out_width, y_in_0,out_height, in_ptr,out_width, out_depth);
			top_left_ptr = transform_get_valid_output_location(x_in_0,out_width, y_in_1,out_height, in_ptr,out_width, out_depth);
			bottom_right_ptr = transform_get_valid_output_location(x_in_1,out_width, y_in_0,out_height, in_ptr,out_width, out_depth);
			bottom_left_ptr = transform_get_valid_output_location(x_in_1,out_width, y_in_1,out_height, in_ptr,out_width, out_depth);
			if(top_right_ptr == NULL && top_left_ptr == NULL && bottom_right_ptr == NULL && bottom_left_ptr == NULL){
				continue;
			}
			(*num_interpolation_info)++;
			blerp_info_list_ptr->top_right_ptr=top_right_ptr;
			blerp_info_list_ptr->top_left_ptr=top_left_ptr;
			blerp_info_list_ptr->bottom_right_ptr=bottom_right_ptr;
			blerp_info_list_ptr->bottom_left_ptr=bottom_left_ptr;
			blerp_info_list_ptr->x_frac = x_in_prime - (float)x_in_0;
			blerp_info_list_ptr->y_frac = y_in_prime - (float)y_in_0;
			blerp_info_list_ptr->out_ptr = out_ptr_start + ((y_out * out_width) + x_out)*out_depth;
			blerp_info_list_ptr++;
		}
	}
	nn_sem_post(tdata->donesem);
}

struct tdata_interpolation{
	struct bilinear_interpolation_info *blerp_info_list_ptr;
	int32_t num_interpolation_info;
	int32_t out_depth;
	nn_sem_t *donesem;
};
static void transform_bilinear_interpolate(struct nn_graph *nn,void* arg){
	struct tdata_interpolation * tdata = (struct tdata_interpolation *) arg;
	struct bilinear_interpolation_info * blerp_info_ptr = tdata->blerp_info_list_ptr;
	int32_t out_depth = tdata->out_depth;
	int32_t num_interpolation_info = tdata->num_interpolation_info;
	for (int i = 0; i < num_interpolation_info; i++){
		const float y_frac = blerp_info_ptr->y_frac;
		const float x_frac = blerp_info_ptr->x_frac;
		float * out = blerp_info_ptr->out_ptr;
		const float * top_right_ptr = blerp_info_ptr->top_right_ptr;
		const float * top_left_ptr = blerp_info_ptr->top_left_ptr;
		const float * bottom_right_ptr = blerp_info_ptr->bottom_right_ptr;
		const float * bottom_left_ptr = blerp_info_ptr->bottom_left_ptr;
		for(int z_out = 0; z_out < out_depth; z_out++) {
			const float top_right_data = (top_right_ptr == NULL) ? 0.0f : top_right_ptr[z_out];
			const float top_left_data = (top_left_ptr == NULL) ? 0.0f : top_left_ptr[z_out];
			const float bottom_right_data = (bottom_right_ptr == NULL) ? 0.0f : bottom_right_ptr[z_out];
			const float bottom_left_data = (bottom_left_ptr == NULL) ? 0.0f : bottom_left_ptr[z_out];
			out[z_out] = bilinear_interpolate(top_right_data,bottom_right_data,top_left_data,bottom_left_data,x_frac,y_frac);
		}
		blerp_info_ptr++;
	}
	nn_sem_post(tdata->donesem);
}

struct  tdata_memset{
	void *data;
	int32_t size;
	int val;
	nn_sem_t *donesem;
};
static void transform_memset(struct nn_graph *nn,void* arg){
	struct tdata_memset * tdata = (struct tdata_memset *) arg;
	vmemset_asm(tdata->data, tdata->val, tdata->size);
	nn_sem_post(tdata->donesem);
}

struct  tdata_memcpy{
	void *detached_ptr;
	void *temp_ptr;
	void *contiguous_ptr;
	int32_t copy_size;
	int32_t over_lapping_memory;
	nn_sem_t *donesem;
};
static void transform_memcpy(struct nn_graph *nn,void *arg){
	struct tdata_memcpy * tdata = (struct tdata_memcpy*) arg;
	if (tdata->over_lapping_memory){
		vmemcpy_asm(tdata->temp_ptr, tdata->detached_ptr, tdata->copy_size);
		vmemcpy_asm(tdata->contiguous_ptr, tdata->temp_ptr, tdata->copy_size);
	}else{
		vmemcpy_asm(tdata->contiguous_ptr, tdata->detached_ptr, tdata->copy_size);
	}
	nn_sem_post(tdata->donesem);
}

static int image_transform_execute_f(struct nn_node *self, struct nn_graph *nn){
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *transform_tensor = self->inputs[1];
	// Setup our output_tensors/out_min/out_max properly
	struct tensor *out_tensor = self->outputs[0];
	int out_batch = input_tensor->shape.batches;
	int out_width = input_tensor->shape.width;
	int out_height = input_tensor->shape.height;
	int out_depth = input_tensor->shape.depth;
	tensor_set_shape(out_tensor, out_batch, out_height, out_width, out_depth);
	int elements = out_batch * out_width * out_height * out_depth;

	int halved_output_height = (out_height+1)/2;
	out_tensor->data_size = elements * sizeof(float);
	if (out_tensor->data_size > out_tensor->max_size) {
		return errlog(nn, "out too small");
	}
	int32_t batch_transform_offset;
	float a0,a1,a2,b0,b1,b2,c0,c1;
	float * input = (float *) input_tensor->data;
	const int32_t image_size = out_width*out_height*out_depth;
	float * out = (float*)out_tensor->data;
	int precalc_scratch_size_per_thread = out_width*halved_output_height*sizeof(struct bilinear_interpolation_info);
	if (nn_scratch_grow(nn,precalc_scratch_size_per_thread*3)){
		return errlog(nn,"oops scratch");
	}
	int32_t num_interpolate_info_thread1;
	int32_t num_interpolate_info_thread2;
	int32_t size_interpolate_info_thread1;
	int32_t size_interpolate_info_thread2;

	nn_sem_t sem_thread1;
	nn_sem_t sem_thread2;

	struct tdata_memset memset_worker ={
		.data = out_tensor->data,
		.size = out_tensor->data_size,
		.val = 0,
		.donesem = &sem_thread1
	};
	nn_sem_init(&sem_thread1,0);
	nn_os_work_for_vector(nn,transform_memset,&memset_worker);
	nn_sem_wait(&sem_thread1);


	// out_tensor->data_size=0;
	for (int n = 0; n < out_batch; n++) {
		batch_transform_offset = n * 8;
		a0 = tensor_get_float(transform_tensor,batch_transform_offset+0);
		a1 = tensor_get_float(transform_tensor,batch_transform_offset+1);
		a2 = tensor_get_float(transform_tensor,batch_transform_offset+2);
		b0 = tensor_get_float(transform_tensor,batch_transform_offset+3);
		b1 = tensor_get_float(transform_tensor,batch_transform_offset+4);
		b2 = tensor_get_float(transform_tensor,batch_transform_offset+5);
		c0 = tensor_get_float(transform_tensor,batch_transform_offset+6);
		c1 = tensor_get_float(transform_tensor,batch_transform_offset+7);
		num_interpolate_info_thread1 = 0;
		num_interpolate_info_thread2 = 0;

		struct tdata_precalculation precalculation_worker1 = 
		{
		.a0 = a0,	
		.a1 = a1,	
		.a2 = a2,	
		.b0 = b0,	
		.b1 = b1,	
		.b2 = b2,	
		.c0 = c0,	
		.c1 = c1,	
		.num_interpolation_info = &num_interpolate_info_thread1,
		.blerp_info_list_ptr = nn->scratch,
		.out_width = out_width,
		.out_height = out_height,
		.out_depth = out_depth,
		.y_out_start = 0,
		.y_out_end = halved_output_height,
		.in_ptr = input,
		.out_ptr_start = out,
		.donesem = &sem_thread1,
		};
		struct tdata_precalculation precalculation_worker2 = 
		{
		.a0 = a0,	
		.a1 = a1,	
		.a2 = a2,	
		.b0 = b0,	
		.b1 = b1,	
		.b2 = b2,	
		.c0 = c0,	
		.c1 = c1,	
		.num_interpolation_info = &num_interpolate_info_thread2,
		.blerp_info_list_ptr = (struct bilinear_interpolation_info *)
								((uint8_t*)nn->scratch + precalc_scratch_size_per_thread),
		.out_width = out_width,
		.out_height = out_height,
		.out_depth = out_depth,
		.y_out_start = halved_output_height,
		.y_out_end = out_height,
		.in_ptr = input,
		.out_ptr_start = out,
		.donesem = &sem_thread2,
		};
		nn_sem_init(&sem_thread1,0);
		nn_sem_init(&sem_thread2,0);
		nn_os_work_for_vector(nn,transform_precalculate_data_location,&precalculation_worker1);
		nn_os_work_for_vector(nn,transform_precalculate_data_location,&precalculation_worker2);

		nn_sem_wait(&sem_thread1);
		nn_sem_wait(&sem_thread2);

		size_interpolate_info_thread1 = num_interpolate_info_thread1* 
											sizeof(struct bilinear_interpolation_info);
		size_interpolate_info_thread2 = num_interpolate_info_thread2* 
											sizeof(struct bilinear_interpolation_info);
		//make the interpolation data contiguous to help with loadbalanced interpolation
		if(size_interpolate_info_thread1 < precalc_scratch_size_per_thread){
			struct tdata_memcpy tmcp =
			{
			.detached_ptr = (uint8_t*)nn->scratch + precalc_scratch_size_per_thread,
			.temp_ptr = (uint8_t*)nn->scratch + precalc_scratch_size_per_thread * 2,
			.contiguous_ptr = (uint8_t*)nn->scratch + size_interpolate_info_thread1,
			.copy_size =  size_interpolate_info_thread2,
			.over_lapping_memory = ((size_interpolate_info_thread2 +size_interpolate_info_thread1) >
										 precalc_scratch_size_per_thread) ? 1 : 0,
			.donesem = &sem_thread1,
			};
			nn_sem_init(&sem_thread1,0);
			nn_os_work_for_vector(nn,transform_memcpy,&tmcp);
			nn_sem_wait(&sem_thread1);
		}
		//try to perform equal number of interpolation across both threads
		int32_t num_interpolate_total = (num_interpolate_info_thread1 + num_interpolate_info_thread2);
		num_interpolate_info_thread1 = (num_interpolate_total + 1) / 2;
		num_interpolate_info_thread2 = num_interpolate_total - num_interpolate_info_thread1;
		size_interpolate_info_thread1 = num_interpolate_info_thread1 * sizeof(struct bilinear_interpolation_info);

		struct tdata_interpolation interpolation_worker1 = {
		.blerp_info_list_ptr = nn->scratch,
		.num_interpolation_info = num_interpolate_info_thread1,
		.out_depth = out_depth,
		.donesem = &sem_thread1,
		};
		struct tdata_interpolation interpolation_worker2 = {
		.blerp_info_list_ptr = (struct bilinear_interpolation_info*)
								((uint8_t*)nn->scratch + size_interpolate_info_thread1),
		.num_interpolation_info = num_interpolate_info_thread2,
		.out_depth = out_depth,
		.donesem = &sem_thread2,
		};
		nn_sem_init(&sem_thread1,0);
		nn_sem_init(&sem_thread2,0);
		nn_os_work_for_vector(nn,transform_bilinear_interpolate, &interpolation_worker1);
		nn_os_work_for_vector(nn,transform_bilinear_interpolate, &interpolation_worker2);
		nn_sem_wait(&sem_thread1);
		nn_sem_wait(&sem_thread2);
		out += image_size;
		input += image_size;
	}

	return 0;
}


struct nn_node_ops nn_ops_for_ImageTransform_f = {
		.execute = image_transform_execute_f,
		.check = NULL,
		.ctor = node_alloc_common,
		.dtor = node_free_common,
		.n_inputs = NN_IOCOUNT(2),
		.n_outputs = NN_IOCOUNT(1),
		.flags = NN_NODE_FLAG_CLS_IMAGETRANSFORM,
};
