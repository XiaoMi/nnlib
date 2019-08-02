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
#include <quantize.h>
#include <math.h>

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

#define MIN(A,B) ( A < B ? A : B)
#define MT
#ifdef MT
struct resize_8_t_data {
	int whoami;
	const uint8_t *in_data;
	uint32_t b, h_in, w_in, d_in;
	uint8_t *out_data;
	uint32_t newwidth;
	uint32_t newheight;
	nn_sem_t donesem;
	uint32_t start_row;
	uint32_t rows;
	float xscale;
	float yscale;
};


// Specific method for align corners.  So that performance of non-aligned isn't impacted

// TODO: Merge these without impacting non-aligned corners
static void resize8_worker_thread_align_corners(struct nn_graph *nn, void *tdata) {
	struct resize_8_t_data *td = tdata;
	uint8_t *out = td->out_data;
	uint8_t *hstart;
	uint8_t *wstart;
	uint8_t *bstart;
	uint32_t close_h;
	uint32_t close_w;
	uint32_t depth_bytes = td->d_in * sizeof(uint8_t);
	int b, h,w;

 	for (b = 0; b < td->b; b++) {
 		out = td->out_data + b * td->newwidth * td->newheight * td->d_in;
 		out += td->start_row * td->newwidth * td->d_in;
 		bstart = (uint8_t*)td->in_data + b*td->h_in*td->w_in*td->d_in;
 		for (h = td->start_row; h < (td->start_row + td->rows); h++) {
		 	close_h = MIN(round(h*td->yscale), td->h_in - 1);
 			hstart = bstart + close_h*td->w_in*td->d_in;
 			for (w = 0; w < td->newwidth; w++) {
 				close_w = MIN(round(w*td->xscale), td->w_in -1);
 				wstart = hstart + close_w*td->d_in;
 				vmemcpy_asm(out,wstart,depth_bytes);
 				out += depth_bytes;
 			}
 		}
 	}
	nn_sem_post(&td->donesem);
}

static void resize8_worker_thread(struct nn_graph *nn, void *tdata) {
	struct resize_8_t_data *td = tdata;
	uint8_t *out = td->out_data;
	uint8_t *hstart;
	uint8_t *wstart;
	uint8_t *bstart;
	uint32_t close_h;
	uint32_t close_w;
	uint32_t depth_bytes = td->d_in * sizeof(uint8_t);
	int b, h,w;

 	for (b = 0; b < td->b; b++) {
 		out = td->out_data + b * td->newwidth * td->newheight * td->d_in;
 		out += td->start_row * td->newwidth * td->d_in;
 		bstart = (uint8_t*)td->in_data + b*td->h_in*td->w_in*td->d_in;
 		for (h = td->start_row; h < (td->start_row + td->rows); h++) {
 			close_h = h * td->yscale;
 			hstart = bstart + close_h*td->w_in*td->d_in;
 			for (w = 0; w < td->newwidth; w++) {
 				close_w = w*td->xscale;
 				wstart = hstart + close_w*td->d_in;
 				vmemcpy_asm(out,wstart,depth_bytes);
 				out += depth_bytes;
 			}
 		}
 	}
	nn_sem_post(&td->donesem);
}

static int resizenear_8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	uint32_t align_corners = 0;
	if (self->n_inputs == 5) {
		const struct tensor *align_corners_tensor = self->inputs[4];
		align_corners = tensor_get_int32(align_corners_tensor, 0);
	}
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	const int32_t *newdims = newdim_tensor->data;
	const int32_t newheight = newdims[0];
	const int32_t newwidth = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	float xscale;
	float yscale;
	uint8_t *out = out_tensor->data;
	const uint8_t *in = in_tensor->data;

	if(  tensor_out_prepare_normal( out_tensor, b_in,newheight,newwidth,d_in, NN_TYPE_UINT8)!= 0 ){
		return errlog(nn,"output prepare failed");
	}
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	struct resize_8_t_data tdata[NUM_THREADS];
	uint32_t rows_per_thread = newheight / NUM_THREADS;

	if (align_corners && (newheight > 1) && (newwidth > 1)) {
		xscale = (float)(w_in-1)/((float)newwidth-1);
		yscale = (float)(h_in-1)/((float)newheight-1);
	} else {
		xscale = (float)w_in/newwidth;
		yscale = (float)h_in/newheight;
	}
	for (int tid = 0; tid < NUM_THREADS; tid++) {
			tdata[tid].whoami = tid;
			tdata[tid].in_data = in;
			tdata[tid].b = b_in;
			tdata[tid].h_in = h_in;
			tdata[tid].w_in = w_in;
			tdata[tid].d_in = d_in;
			tdata[tid].out_data = out;
			tdata[tid].newwidth = newwidth;
			tdata[tid].newheight = newheight;
			tdata[tid].xscale = xscale;
			tdata[tid].yscale = yscale;

			tdata[tid].start_row = rows_per_thread * tid;
			tdata[tid].rows = rows_per_thread;
			nn_sem_init(&tdata[tid].donesem, 0);
		}
	tdata[NUM_THREADS-1].rows += newheight % NUM_THREADS;

	// Use a different method depending on align_corners.  This is to keep
	// non-aligned corners fast.
	for(int i = 0; i < NUM_THREADS; i++){
		if (align_corners) {
        	nn_os_work_for_vector(nn, resize8_worker_thread_align_corners, &tdata[i]);
        } else {
        	nn_os_work_for_vector(nn, resize8_worker_thread, &tdata[i]);
        }
    }
    for (int i=0; i<NUM_THREADS; i++) {
        nn_sem_wait(&tdata[i].donesem);
    }
	return 0;
}
#else
static int resizenear_8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	uint32_t align_corners = 0;
	if (self->n_inputs == 5) {
		const struct tensor *align_corners_tensor = self->inputs[4];
		align_corners = tensor_get_int32(align_corners_tensor, 0);
	}
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	const int32_t *newdims = newdim_tensor->data;
	const int32_t newheight = newdims[0];
	const int32_t newwidth = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	float xscale;
	float yscale;
	uint32_t close_h;
	uint32_t close_w;
	int b,h,w;
	char *out = out_tensor->data;
	const uint8_t *in = in_tensor->data;
	const uint8_t *bstart;
	const uint8_t *hstart;
	const uint8_t *wstart;
	uint32_t depth_bytes = d_in * sizeof(uint8_t);
	uint32_t total_bytes = b_in*newheight*newwidth*depth_bytes;

	if (total_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,b_in,newheight,newwidth,d_in);
	out_tensor->data_size = total_bytes;
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	if (align_corners && (newheight > 1) && (newwidth > 1)) {
		xscale = (float)(w_in-1)/((float)newwidth-1);
		yscale = (float)(h_in-1)/((float)newheight-1);
	} else {
		xscale = (float)w_in/newwidth;
		yscale = (float)h_in/newheight;
	}

	for (b = 0; b < b_in; b++) {
		bstart = in + b*h_in*w_in*d_in;
		for (h = 0; h < newheight; h++) {
			if (align_corners) {
				close_h = MIN(round(h*yscale), h_in - 1);
			} else {
				close_h = h * yscale;
			}
			hstart = bstart + close_h*w_in*d_in;
			for (w = 0; w < newwidth; w++) {
				if (align_corners) {
					close_w = MIN(round(w*xscale), w_in - 1);
				} else {
					close_w = w*xscale;
				}
				wstart = hstart + close_w*d_in;
				vmemcpy_asm(out,wstart,depth_bytes);
				out += depth_bytes;
			}
		}
	}
	return 0;
}
#endif
static int resizenear_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *newdims = newdim_tensor->data;
	const int32_t newheight = newdims[0];
	const int32_t newwidth = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	const float xscale = (float)w_in/newwidth;
	const float yscale = (float)h_in/newheight;
	uint32_t close_h;
	uint32_t close_w;
	int b,h,w;
	char *out = out_tensor->data;
	const float *in = in_tensor->data;
	const float *bstart;
	const float *hstart;
	const float *wstart;
	uint32_t depth_bytes = d_in * sizeof(float);
	uint32_t total_bytes = b_in*newheight*newwidth*depth_bytes;

	if (total_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,b_in,newheight,newwidth,d_in);
	out_tensor->data_size = total_bytes;

	for (b = 0; b < b_in; b++) {
		bstart = in + b*h_in*w_in*d_in;
		for (h = 0; h < newheight; h++) {
			close_h = h * yscale;
			hstart = bstart + close_h*w_in*d_in;
			for (w = 0; w < newwidth; w++) {
				close_w = w*xscale;
				wstart = hstart + close_w*d_in;
				memcpy(out,wstart,depth_bytes);
				out += depth_bytes;
			}
		}
	}
	return 0;
}



struct nn_node_ops nn_ops_for_ResizeNearestNeighbor_f = {
	.execute = resizenear_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),		// TODO support align corners input
	.n_outputs = NN_IOCOUNT(1),
};

// INS: intensor, inmin, inmax, newdims, align_corners (optional)
// OUTS: outputtensor, outmax, outmin

struct nn_node_ops nn_ops_for_ResizeNearestNeighbor_8 = {
	.execute = resizenear_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
};

