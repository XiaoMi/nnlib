
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
 * Given a start offset and width for each dimention in the input tensor,
 * create a new output tensor with just the slice specified.
 * 
 */
/*
 *  inputs:
 *     0 - data tensor
 *     1 - start spec (see below)
 *     2 - size spec (see below)
 *     3,4 (if quantized): input min,max
 *
 *     The 'start spec' and 'size spec' are both tensors of shape (1,1,1,k)
 *     where k = 1..4.
 *     e.g. you can have
 *         start_spec [1,1,1,4] = { b0, h0, w0, d0 }
 *         size_spec [1,1,1,4] = { b_size, h_size, w_size, d_size }
 *
 *    And then the slice will be  [b0 ... b0+b_size-1] on batches, etc.
 *
 *    - For k <= 4, both 'start spec' and 'size spec' must still have the same shape,
 *     and the missing dimensions are dropped on the left; e.g.
 *         start_spec [1,1,1,2] = { w0, d0 }
 *         size_spec [1,1,1,2] = { w_size, d_size }
 *       .. and the 'batch' and 'height' dimensions are not sliced.
 *     Note:
 *         a 'size' of -1 means all available  (so start = 0, size=-1 means retain all)

 */
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>

#ifdef HEXAGON_V66
#define SLICE_MAX_THREADS 4
#else
#define SLICE_MAX_THREADS 2
#endif

// a 'job' is a 2d mem copy.
// if height == 1, it can be done as a normal memcpy.
// The input addresses are expressed as an offset from start of
// tensor, in case the input moves from run to run.
typedef struct
{
	void* dst;
	unsigned src_offs;		// 'src' is relative to info->input_tensor_base
	int width, height;
	int dst_stride, src_stride;

} slice_job_info;


typedef struct
{
	slice_job_info* jobs;
	int jobs_len;
	int num_threads;
	uint8_t const * input_tensor_base;		// the input tensor location.
	nn_sem_t done_sem;
} slice_thread_info;

static void slice_thread_work(struct nn_graph *nn, void *work_info) {
	slice_thread_info* info = work_info;
	slice_job_info* job_ptr = info->jobs;
	Q6_dcfetch_A(job_ptr);
	int jobs_len = info->jobs_len;
	uint8_t const * in_base = info->input_tensor_base;

	for(int i = 0; i < jobs_len; i++){
		Q6_dcfetch_A(&job_ptr[i+1]);
		slice_job_info const * jp = &job_ptr[i];
		int ht = jp->height;
		uint8_t const * srcp = in_base + jp->src_offs;
		if( ht == 1){
			vmemcpy_asm( jp->dst, srcp, jp->width);
		}else{
			vmemcpy_2d_general_asm(jp->width, ht, jp->dst, jp->dst_stride, srcp, jp->src_stride );
		}
	}
	nn_sem_post(&info->done_sem);
}

static int slice_prepare(struct nn_node *self, struct nn_graph *nn, int element_size, int typeid) {
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *start_tensor = self->inputs[1];
	const struct tensor *size_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	int b,b_in,b_start,b_size;
	int h,h_in,h_start,h_size;
	int w_in,w_start,w_size;
	int d_in,d_start,d_size;

	logmsg(nn,2,"slice node %p preparation",self);

	int nspec = start_tensor->shape.depth;
	if( start_tensor->shape.batches != 1 || start_tensor->shape.height != 1
		|| start_tensor->shape.width != 1 || nspec < 1 || nspec > 4
		|| !shape_matches( &start_tensor->shape, &size_tensor->shape) ){
		return errlog(nn,"bad size/shape spec for 'slice'");
	}

	b_in = input_tensor->shape.batches;
	h_in = input_tensor->shape.height;
	w_in = input_tensor->shape.width;
	d_in = input_tensor->shape.depth;

	
	logmsg(nn,2,"shape of in/start/size: b: %d/%d/%d h: %d/%d/%d w: %d/%d/%d d: %d/%d/%d",
		b_in,start_tensor->shape.batches,size_tensor->shape.batches,
		h_in,start_tensor->shape.height,size_tensor->shape.height,
		w_in,start_tensor->shape.width,size_tensor->shape.width,
		d_in,start_tensor->shape.depth,size_tensor->shape.depth);

	b_start = 0;	b_size = -1;
	h_start = 0; 	h_size = -1;
	w_start = 0;	w_size = -1;
	d_start = tensor_get_int32( start_tensor, nspec-1);
	d_size = tensor_get_int32( size_tensor, nspec-1);

	if( nspec >= 2 ){
		w_start = tensor_get_int32(start_tensor,nspec-2);
		w_size = tensor_get_int32(size_tensor,nspec-2);
		if( nspec >= 3 ){
			h_start = tensor_get_int32(start_tensor,nspec-3);
			h_size = tensor_get_int32(size_tensor,nspec-3);
			if( nspec >= 4){
				b_start = tensor_get_int32(start_tensor,0);
				b_size = tensor_get_int32(size_tensor,0);
			}
		}
	}

	if (b_size == -1) b_size = b_in - b_start;
	if (h_size == -1) h_size = h_in - h_start;
	if (w_size == -1) w_size = w_in - w_start;
	if (d_size == -1) d_size = d_in - d_start;


	logmsg(nn,2,"in/start/size: b: %d/%d/%d h: %d/%d/%d w: %d/%d/%d d: %d/%d/%d order_skip=%d",
		b_in,b_start,b_size,
		h_in,h_start,h_size,
		w_in,w_start,w_size,
		d_in,d_start,d_size,
		4-nspec);

	if (b_size <= 0) return errlog(nn,"bad b_size");
	if (h_size <= 0) return errlog(nn,"bad h_size");
	if (w_size <= 0) return errlog(nn,"bad w_size");
	if (d_size <= 0) return errlog(nn,"bad d_size");
	if ((b_start+b_size) > b_in) return errlog(nn,"in b too small");
	if ((h_start+h_size) > h_in) return errlog(nn,"in h too small");
	if ((w_start+w_size) > w_in) return errlog(nn,"in w too small");
	if ((d_start+d_size) > d_in) return errlog(nn,"in d too small");

	if( tensor_out_prepare_normal( out_tensor, b_size,h_size,w_size,d_size, typeid ) !=  0) {
		return errlog(nn,"out too small");
	}

	// source addresses are calculated as offsets from the start of tensor, in case it moves from run to run.
	// skip the 'offset'
	unsigned indata_offset= element_size*( d_start + d_in * (w_start + w_in * (h_start + h_in*b_start)));

	char *out = out_tensor->data;

	// try to simplify the layout, move things into inner loops.
	// (for now, all strides are in elements, not bytes).
	//
	struct ddesc { int32_t siz, in_stride; }
	layout[4] = {
			// dimens.   in_stride
			{   d_size,   1, },
			{   w_size,   d_in },
			{   h_size,   w_in*d_in},
			{   b_size,   h_in*w_in*d_in },
	};
	// if a dimension's input stride is the product of the previous dim's size and in stride,
	// then they can be merged into one with the lower stride, and the product of sizes.
	// The other dims are moved down and a size=1 dim added at the outside.
	// layout[0].in_stride will remain = 1.
	//
	int ndims = 4;
	for( int i = 0; i < ndims-1; ){
		if( layout[i+1].in_stride == layout[i].siz * layout[i].in_stride ){
			layout[i].siz *= layout[i+1].siz;
			for( int j = i+1; j < ndims-1; j++){
				layout[j] = layout[j+1];		// squeeze others down.
			}
			ndims --;
			layout[ndims].siz = 1;		// and stride doesn't matter
		}else{
			i++;
		}
	}
	// @@ Is it still possible to have e.g. w_size = 1, h_size > 1? if so
	// we should squeeze out the size=1 dims to get better 2d-memcpy.
	int b_in_stride, h_in_stride, w_in_stride;

	d_size = layout[0].siz;
	w_size = layout[1].siz;
	w_in_stride = layout[1].in_stride * element_size;
	h_size = layout[2].siz;
	h_in_stride = layout[2].in_stride * element_size;
	b_size = layout[3].siz;
	b_in_stride = layout[3].in_stride * element_size;

	// w_size * d_size is handled by 2d-memcpy, so this is the
	// # of 2d-memcpys we need.
	//
	int workloads_len = b_size * h_size;
	// many use cases become a single memcpy_2d.
	// We split these in two to make two work units (unless w_size=1 in which
	// case it's just a single memcpy)
	// For instance, if you have a tensor [2,20,30,128] and you want to trim to to [2,20,30,120]
	// that will be w_size = 2*20*30, d_size = 120; we want to split that in two on alternating w's;
	// the dest strides in each job will be 240 rather than 120.
	//
	// TODO: could also split the w=1 cases if d_size*element_size is really big.
	//
	int split_w = 0;
	if( workloads_len == 1 && w_size > 1){
		split_w = 1;
		workloads_len = 2;
	}


	logmsg(nn,2,"total memcpy blocks are %d",workloads_len);
	int num_threads = min_i32(workloads_len, SLICE_MAX_THREADS);
	int workload_for_worker[SLICE_MAX_THREADS];
	int average_workload_per_worker = workloads_len / num_threads;
	int extra_work = workloads_len % num_threads;
	for(int i = 0; i < num_threads; i++){
		workload_for_worker[i] = average_workload_per_worker;
	}
	for(int i = 0; i < extra_work; i++){
		workload_for_worker[i] += 1;
	}

	//calc the memcpy job positions
	int needed_buffer_size = sizeof(slice_job_info) * workloads_len + sizeof(slice_thread_info) * num_threads;
	void *job_buffer = nn_calloc(needed_buffer_size,1);
	self->opaque = job_buffer;
	if(job_buffer ==NULL) {
		return errlog(nn, "can't allocate %d bytes for job buffer", needed_buffer_size);
	}

	slice_thread_info* thrinfo = (slice_thread_info*)job_buffer;
	slice_job_info* jobs_ptr = (slice_job_info*)((uint8_t*)job_buffer + sizeof(slice_thread_info) * num_threads);
	int offset = 0;
	for(int i = 0; i < num_threads; i++)
	{
		thrinfo[i].num_threads = num_threads;
		thrinfo[i].jobs_len = workload_for_worker[i];
		thrinfo[i].jobs = jobs_ptr + offset;
		offset += workload_for_worker[i];
	}
	int d_bytes= d_size* element_size;
	int w_out_stride = d_bytes;
	int h_out_stride = d_bytes * w_size;
	int b_out_stride= h_out_stride * w_size;


	if( split_w){
		// special case, b_size * h_size ==1 and w_size >= 2.
		// normally this would be one big 2d copy; but instead do one for even w's and one for odd.
		int w_odd = w_size>>1;		// # of odd w's
		int w_even = w_size-w_odd;		// # of even
		jobs_ptr[0].dst = out;
		jobs_ptr[0].src_offs= indata_offset;
		jobs_ptr[0].width = d_bytes;
		jobs_ptr[0].height = w_even;
		jobs_ptr[0].src_stride = w_in_stride*2;
		jobs_ptr[0].dst_stride = w_out_stride*2;

		jobs_ptr[1].dst = out + w_out_stride;
		jobs_ptr[1].src_offs = indata_offset+ w_in_stride;
		jobs_ptr[1].width = d_bytes;
		jobs_ptr[1].height = w_odd;
		jobs_ptr[1].src_stride = w_in_stride*2;
		jobs_ptr[1].dst_stride = w_out_stride*2;
	}else{
		// we could swap h_out_stride <-> w_out_stride, w_in_stride <-> h_in_stride, and h_size <-> w_size here
		// if h_size is significantly larger than w_size - to get a smaller # of larger ops.
		// But this reduces the number of jobs, so decision should be made back before the # jobs is found
		for (b = 0; b < b_size; b++) {
			char * out_b = out + b*b_out_stride;
			for (h = 0; h < h_size; h++) {
				jobs_ptr->dst = out_b;
				jobs_ptr->src_offs = indata_offset + b_in_stride * b + h_in_stride * h;
				jobs_ptr->width = d_bytes;
				jobs_ptr->height = w_size;
				jobs_ptr->src_stride = w_in_stride;
				jobs_ptr->dst_stride = w_out_stride;

				out_b += h_out_stride;
				jobs_ptr++;
				/*
				 * Original inner-loop for 1d copies
				for (w = 0; w < w_size; w++) {
					jobs_ptr->dst = out;
					jobs_ptr->src_offs = indata_offset + b_in_stride * b + h_in_stride * h + w_in_stride * w;
					jobs_ptr->size = d_size * element_size;
					out += d_size * element_size;
					jobs_ptr++;
				}
				*/
			}
		}
	}
	return 0;
}

static int slice_run(struct nn_node *self, struct nn_graph *nn) {
	if (self->n_inputs == 5) {
		tensor_copy(self->outputs[1],self->inputs[3]);
		tensor_copy(self->outputs[2],self->inputs[4]);
	}
	slice_thread_info* thrinfo = (slice_thread_info*)self->opaque;
	if( thrinfo == NULL){
		int elbytes, typecode;
		if( self->node_type == OP_QuantizedSlice_16 ){
			typecode = NN_TYPE_QINT8;
			elbytes = sizeof(int16_t);
		}else{
			typecode = NN_TYPE_QUINT8;
			elbytes = sizeof(uint8_t);
		}
		if( slice_prepare(self,nn,elbytes, typecode)!=0) return -1;
		thrinfo = (slice_thread_info*)self->opaque;
	}
	int num_threads = thrinfo[0].num_threads;
	uint8_t const * in = self->inputs[0]->data;
	for(int i = 0; i < num_threads; i++) {
		nn_sem_init(&thrinfo[i].done_sem, 0);
		thrinfo[i].input_tensor_base = in;
		nn_os_work_for_vector(nn, slice_thread_work, &thrinfo[i]);
	}

	for(int i =0; i < num_threads; i++) {
		nn_sem_wait(&thrinfo[i].done_sem);
	}
	return 0;
}

static int slice_run_f(struct nn_node *self, struct nn_graph *nn) {
	slice_thread_info* thrinfo = (slice_thread_info*)self->opaque;
	if( thrinfo == NULL){
		int elbytes, typecode;
		if( self->node_type == OP_Slice_8 ){
			typecode = NN_TYPE_UINT8;
			elbytes = sizeof(uint8_t);
		}else{
			typecode = (self->node_type == OP_Slice_f)? NN_TYPE_FLOAT: NN_TYPE_INT32;
			elbytes = (self->node_type == OP_Slice_f)? sizeof(float): sizeof(int32_t);
		}
		if( slice_prepare(self,nn,elbytes, typecode)!=0) return -1;
		thrinfo = (slice_thread_info*)self->opaque;
	}
	int num_threads = thrinfo[0].num_threads;
	uint8_t const * in = self->inputs[0]->data;
	for(int i = 0; i < num_threads; i++) {
		nn_sem_init(&thrinfo[i].done_sem, 0);
		thrinfo[i].input_tensor_base = in;
		nn_os_work_for_vector(nn, slice_thread_work, &thrinfo[i]);
	}

	for(int i =0; i < num_threads; i++) {
		nn_sem_wait(&thrinfo[i].done_sem);
	}
	return 0;
}

static int slice_dtor(struct nn_node *self, struct nn_graph *nn) {
	if(self->opaque != NULL) {
		nn_free(self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self, nn);
}

static int prepare_work_f(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor){
	return slice_prepare(self,nn,sizeof(float), NN_TYPE_FLOAT);
}

static int prepare_work_8(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return slice_prepare(self,nn,sizeof(uint8_t), NN_TYPE_UINT8);
}

static int prepare_work_int32(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return slice_prepare(self,nn,sizeof(int32_t), NN_TYPE_INT32);
}

static int prepare_work_q8(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return slice_prepare(self,nn,sizeof(uint8_t), NN_TYPE_QUINT8);
}
static int prepare_work_q16(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return slice_prepare(self,nn,sizeof(int16_t), NN_TYPE_QINT16);
}


struct nn_node_ops nn_ops_for_Slice_f = {
	.execute = slice_run_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = slice_dtor,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
	.earlywork_note_pred = prepare_work_f,
};

struct nn_node_ops nn_ops_for_Slice_8 = {
	.execute = slice_run_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = slice_dtor,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
	.earlywork_note_pred = prepare_work_8,
};

struct nn_node_ops nn_ops_for_Slice_int32 = {
	.execute = slice_run_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = slice_dtor,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
	.earlywork_note_pred = prepare_work_int32,
};

struct nn_node_ops nn_ops_for_QuantizedSlice_8 = {
	.execute = slice_run,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = slice_dtor,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.earlywork_note_pred = prepare_work_q8,
};


struct nn_node_ops nn_ops_for_QuantizedSlice_16 = {
	.execute = slice_run,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = slice_dtor,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.earlywork_note_pred = prepare_work_q16,
};

