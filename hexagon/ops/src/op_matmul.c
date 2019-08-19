
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
 * This contains the code for matrix multiply op
 */

#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>
#include <quantize.h>
#include <hvx_inlines.h>
#ifndef __hexagon__
#include <malloc.h>
#endif
#define ALIGN_SIZE 128

#define MAXMUL_THREADS 2

/* 8x8 matrix multiply --> 32 bits */

struct tdata {
	struct nn_node *self;
	void (*f)(struct nn_node *self, struct nn_graph *nn, int32_t a_off, int32_t b_off, uint32_t tid);
	int32_t retval;
	uint32_t tid;
	nn_sem_t *donesem;
};

static inline int matmul_execute(struct nn_node *self, struct nn_graph *nn,
		void (*f)(struct nn_node *self, struct nn_graph *nn, int32_t a_offset, int32_t b_offset, uint32_t tid), uint32_t tid)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *min_a_tensor = self->inputs[2];
	const struct tensor *max_a_tensor = self->inputs[3];
	const struct tensor *min_b_tensor = self->inputs[4];
	const struct tensor *max_b_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	uint32_t a_batches = a_tensor->shape.batches;
	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_height = a_tensor->shape.height;
	uint32_t a_depth = a_tensor->shape.depth;

	uint32_t b_batches = b_tensor->shape.batches;
	uint32_t b_width = b_tensor->shape.width;
	uint32_t b_height = b_tensor->shape.height;
	uint32_t b_depth = b_tensor->shape.depth;

	uint32_t out_batches = 1;
	uint32_t out_height = 1;
	uint32_t dotprod_len = b_width;
	uint32_t out_width = (a_batches*a_height*a_width*a_depth)/dotprod_len;
	uint32_t out_depth = b_depth;

	float a_max_float = tensor_get_float(max_a_tensor,0);
	float a_min_float = tensor_get_float(min_a_tensor,0);
	float b_max_float = tensor_get_float(max_b_tensor,0);
	float b_min_float = tensor_get_float(min_b_tensor,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float a_level_size = (a_max_float - a_min_float) / 255.0f;
	float b_level_size = (b_max_float - b_min_float) / 255.0f;
	float out_level_size = a_level_size * b_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t a_offset = quantize_uint8(0.0f,a_min_float,a_max_float);
	int32_t b_offset = quantize_uint8(0.0f,b_min_float,b_max_float);

	logmsg(nn,2,"matmul execute. self=%p",self);
	logmsg(nn,2,"matmul in dims: %lux%lux%lux%lu * %lux%lux%lux%lu",
		a_batches,a_height,a_width,a_depth,
		b_batches,b_height,b_width,b_depth);
	logmsg(nn,2,"reshaping A to %lux%lu",out_width,dotprod_len);
	logmsg(nn,2,"matmul out dims: %lux%lux%lux%lu",
			out_batches,out_height,out_width,out_depth);
	//if (a_height != 1  || b_height != 1) return errlog(nn,"oops, height != 1");
	if (b_height != 1) return errlog(nn,"fixme: support B height?");
	if (b_batches != 1) return errlog(nn,"fixme: support B batches");


	if( tensor_out_prepare_normal( out_tensor,out_batches,out_height,out_width,out_depth , NN_TYPE_INT32)!= 0){
		return errlog(nn,"output too small");
	}

	tensor_set_single_float( out_min, out_min_val);
	tensor_set_single_float( out_max, out_max_val);

	f(self, nn, a_offset, b_offset, tid);

	logmsg(nn,2,"matmul execute done!");
	return 0;
}

static void matmul_worker(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	td->retval = matmul_execute(td->self,nn,td->f, td->tid);
	nn_sem_post(td->donesem);
}

static int matmul_launch(struct nn_node *self, struct nn_graph *nn,
		void (*f)(struct nn_node *self, struct nn_graph *nn, int32_t a_offset, int32_t b_offset, uint32_t tid))
{
	struct tdata td[MAXMUL_THREADS];
	nn_sem_t sem;

	nn_scratch_reset(nn);

	nn_sem_init(&sem,0);
	for (int32_t i = 0; i < MAXMUL_THREADS; i++) {
		td[i].f = f;
		td[i].self = self;
		td[i].tid = i;
		td[i].retval = 0;
		td[i].donesem = &sem;
		nn_os_work_for_vector(nn, matmul_worker, &td[i]);
	}
	int res = 0;
	for (int32_t i = 0; i < MAXMUL_THREADS; i++) {
		nn_sem_wait(&sem);
		res |= td[i].retval;
	}
	return res;
}

static inline void matmul_ref(
		struct nn_node *self,
		struct nn_graph *nn,
		int32_t a_offset,
		int32_t b_offset,
		uint32_t tid
	)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint8_t *a = a_tensor->data;
	uint8_t *b = b_tensor->data;
	int32_t *out = out_tensor->data;

	int32_t adata;
	int32_t bdata;
	int32_t sum;
	int32_t x;
	int32_t y;
	int32_t i;

	uint32_t b_width = b_tensor->shape.width;
	uint32_t b_depth = b_tensor->shape.depth;
	uint32_t out_width = (a_tensor->shape.batches*a_tensor->shape.height*a_tensor->shape.width*a_tensor->shape.depth)/b_width;
	uint32_t out_depth = b_depth;

	if( tid != 0)
		return;

    logmsg(nn,2,"a_widthxa_depth=%lux%lu a_offset=%ld b_offset=%ld",
    		out_width, b_width, a_offset, b_offset);
	for (y = 0; y < out_width; y++) {
		for (x = 0; x < out_depth; x++) {
			sum = 0;
			for (i = 0; i < b_width; i++) {
				adata = a[i+y*b_width] - a_offset;
				bdata = b[x+i*b_depth] - b_offset;
				sum += adata * bdata;
			}
			out[x+y*out_depth] = sum;
		}
	}
	logmsg(nn,2,"matmul execute ref done!");
}

static inline void matmul_asm(
		struct nn_node *self,
		struct nn_graph *nn,
		int32_t a_offset,
		int32_t b_offset,
		uint32_t tid
		)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint8_t *a = a_tensor->data;
	int32_t *out = out_tensor->data;

	uint32_t a_depth = b_tensor->shape.width;
	uint32_t b_depth = b_tensor->shape.depth;
	uint32_t a_batches = (a_tensor->shape.batches*a_tensor->shape.width*a_tensor->shape.height*a_tensor->shape.depth)/a_depth;
	uint32_t out_depth = b_depth;
	int32_t i;

	//int b_depth_pad = (b_depth + 32-1)&~(32-1);
	int a_depth_pad = (a_depth + 16-1)&~(16-1);

	// ASM code may be used if a_batches ==1, or if a_depth is a multiple of 16.
	// (since we need to keep aligned by 16 on a_ptr).
	// If the output depth is not a multiple of 32, some batches will have misaligned output addresses.
	// Each time we start a new batch, if the output address isn't aligned, we compute it to an aligned temp
	// and copy those to the output using vmemcpy.
	// When multiple threads are in use, we also need to do this when the *end* of the depth chunk is
	// misaligned, to avoid over-writing the start of the next batch -- see comment below and 'all_misaligned'.
	//

	uint8_t const * bptr = (uint8_t const *)self->opaque;		// read 'b side' from here...
	//
	//if (a_batches == 1 || a_depth == a_depth_pad )	// can use hvx
	if( a_batches==1 || a_depth == a_depth_pad)
	{
		logmsg(nn,2,"Pad A: a_widthxa_depth=%lux%lu,a_widthxa_depth_pad=%lux%d, a_offset=%ld b_offset=%ld",
				a_batches, a_depth, a_batches,a_depth_pad, a_offset, b_offset);
		// depth per thread: divide depth by (32*MAX_THREADS), rounded up; then *32
		//
		int depth_thread = ((out_depth + MAXMUL_THREADS*32 - 1) / (MAXMUL_THREADS*32))<<5;
		int out_depth_start = tid * depth_thread;
		int out_depth_end = min_i32( out_depth_start+depth_thread, out_depth);
		int out_depth_todo = out_depth_end - out_depth_start;
		if( out_depth_end <= out_depth_start)
			return;

		// allocate a work buffer, enough to hold our depth slice (only needed if depth misaligned).
		int32_t * temp_buf = NULL;
		if( ( out_depth & 31)!= 0 ){
			temp_buf = (int32_t *) nn_scratch_alloc( nn, sizeof(int32_t) * ((out_depth_todo+31)&~31u) );
			if( temp_buf == NULL) {
				errlog(nn, "can't alloc scratch for %d ints", (int) out_depth_todo);
				goto use_ref_code ;
			}
		}
		// out_depth_start is always a multiple of 32. If out_depth_end is *not* a multiple
		// of 32, we need to use the temp-copy on all operations, even if the start is aligned;
		// otherwise we could spill over the end of the batch on top of work done by the other
		// thread. Exception: if there is only one active thread, since out_depth <= 32, we don't
		// need to do this (so, thread 0 never needs to); also if a_batches ==1 we don't need this.
		//
		int all_misaligned = (out_depth_end&31)!= 0 && tid != 0 && a_batches > 1;

		for( uint32_t ibatch = 0; ibatch < a_batches; ibatch++){
			uint8_t *aptr = a + a_depth_pad * ibatch;

			l2fetch(aptr, a_depth_pad, a_depth_pad, 1);
			l2fetch(bptr+ out_depth_start*a_depth_pad, a_depth_pad, a_depth_pad, min_i32(32, out_depth_todo));

			int outpos = out_depth * ibatch+ out_depth_start; 	// offset of batch in output area.
			int32_t *optr = out + outpos;		// actual output pos
			int32_t *wptr = optr;				// output for the asm op
			int is_misaligned = (outpos & 31) != 0 || all_misaligned;
			if( is_misaligned)
				wptr = temp_buf;		// if misaligned,store here instead
			// loop through, do 32 outputs at once.
			// if misaligned, on each 4 loops, copy out the temp data.
			for(i = out_depth_start; i < out_depth_end; i+=32) {
					wait_for_l2fetch();

				if ((i+32)< out_depth_end)
						l2fetch(bptr+(i+32)*a_depth_pad, a_depth_pad, a_depth_pad, min_i32(32, out_depth_end -i-32));

				gemvmpybbw_asm(
				    aptr,
					-a_offset,
					bptr+i*a_depth_pad,
					-b_offset,
					(int*) wptr,
					min_i32(32, out_depth_end -i),
					a_depth_pad);
				wptr += 32;
			}
			if( is_misaligned ){	// copy remnant from the buffer
				vmemcpy_asm( optr, temp_buf, out_depth_todo * sizeof(int32_t));
			}				
		}
	}
	else
	{
		if( tid !=0)
			return;
	  use_ref_code:
		matmul_ref(self, nn, a_offset, b_offset, tid);
		logmsg(nn,2,"matmul execute asm does not handle this case, for now use reference C code!");

	}
	logmsg(nn,2,"matmul execute asm done!");
}


static int matmul_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	return matmul_launch(self,nn,matmul_ref);
}

static int matmul_execute_asm(struct nn_node *self, struct nn_graph *nn)
{
	return matmul_launch(self,nn,matmul_asm);
}


static int matmul_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking matmul node %p",self);


#define BPAD 32
#define APAD 16
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	uint32_t filt_batches = filt_tensor->shape.filt_batches;
	uint32_t filt_depth = filt_tensor->shape.filt_depth;
	uint32_t out_depth = filt_batches;
	uint8_t *filt = filt_tensor->data;
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	uint32_t filt_elements = filt_depth;
	uint32_t filt_elements_pad = (filt_elements + APAD - 1) & (~(APAD - 1));
	int out_depth_pad = (out_depth + BPAD - 1) & ~(BPAD-1);
	uint32_t consts_size;
	filt_elements_pad = (filt_elements_pad < 32)?32:filt_elements_pad;
	consts_size = filt_elements_pad * out_depth_pad;
	if (nn_scratch_grow(nn,consts_size)){
		return errlog(nn,"couldn't allocate scratch buffer for const rearrangement");
	}
	if (self->opaque == NULL) {
		if ((self->opaque = nn_memalign(ALIGN_SIZE,consts_size)) == NULL) {
			return errlog(nn,"couldn't allocate buffer for const rearrangement");
		}
	}
	nn_scratch_grow(nn,filt_elements_pad*out_depth_pad+256);
	logmsg(nn,2,"Pad B: filt_elements=%lu %lu,out_depth=%lu %d, filt_offset=%ld", filt_elements, out_depth, filt_elements_pad,out_depth_pad, filt_offset);

	nn_mutex_lock(&nn->scratch_mutex);
	pad2d(filt,filt_elements,out_depth,nn->scratch,filt_elements_pad,out_depth_pad,filt_offset);
	transpack(nn->scratch,filt_elements_pad,out_depth_pad,self->opaque);
	nn_mutex_unlock(&nn->scratch_mutex);
	logmsg(nn,2,"matmul node %p check OK",self);
	return 0;
}

static int matmul_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque){
		nn_free(self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self,nn);
}


static struct nn_node *matmul_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	logmsg(nn,2,"matmul node id %x ctor",node_id);
	/* FIXME: replace ops pointers with optimized implementations when available */
	return node_alloc_common(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs);
}

struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8to32 = {
	.execute = matmul_execute_asm,
	.check = matmul_check_ref,
	.ctor = matmul_ctor,
	.dtor = matmul_dtor,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8to32_ref = {
	.execute = matmul_execute_ref,
	.check = matmul_check_ref,
	.ctor = matmul_ctor,
	.dtor = matmul_dtor,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
