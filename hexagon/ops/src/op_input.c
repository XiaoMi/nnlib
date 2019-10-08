
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
#include <nn_asm_ops.h>
#include <string.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for an input node.
 */

struct input_info {
	void *allocated_output;
	nn_sem_t donesem;
};
#if 0
static void input_execute_worker(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	int i;
	struct tensor *out;
	const struct tensor *in;
	struct input_info *info = self->opaque;
	/* Copy input tensor to output */
	logmsg(nn,2,"input execute. self=%p ",self);
	for (i = 0; i < self->n_outputs; i++) {
		out = self->outputs[i];
		in = &nn->inputs[i];
		/* Warning! Inputs come in as max_size not data_size! */
		logmsg(nn,9,"in: [%d,%d,%d,%d] size=%d",
			in->shape.batches,
			in->shape.height,
			in->shape.width,
			in->shape.depth,
			in->max_size);
		if (out->max_size < in->max_size) {
			errlog(nn,"out too small (%lu < %lu)", out->max_size, in->max_size);
			out->shape.batches = 0;
			out->shape.height = 0;
			out->shape.width = 0;
			out->shape.depth = 0;
			out->data_size = 0;  // Communicate the failure upward
			return;
		}
		out->shape = in->shape;
		out->format = in->format;
		out->data_size = in->data_size;
		vmemcpy_asm(out->data,in->data,in->max_size);
	}
	nn_sem_post(&info->donesem);
	logmsg(nn,2,"input %d tensors",nn->n_inputs);
}
#endif
static int input_execute_multibatch(struct nn_node *self, struct nn_graph *nn);

// is endptr[delt-1] in a different page from endptr[-1],
// where 'page' is defined by pagesize (which must be a power of 2)?
//
static inline int
page_cross_check( void const *endptr, int delt, unsigned pagesize)
{
	uint8_t const *p =endptr;
	size_t pos0 = (size_t)(p-1);
	size_t pos1 = (size_t)(p-1+delt);
	return ((pos0^pos1) &~(size_t)(pagesize-1))!=0;
}



static int input_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* OPTIMIZE FAST PATH */
	struct input_info *info = self->opaque;

	if (likely(
		(nn->n_inputs == self->n_outputs)
		&& (self->n_outputs == 1)
		&& nn->batchseq.graph_batches == 0
		&& ((((size_t)(nn->inputs[0].data)) & 127) == 0)
		&& !page_cross_check( (uint8_t const*)nn->inputs[0].data+nn->inputs[0].data_size,
				256, 0x1000)
	  )) {
		self->outputs[0]->shape = nn->inputs[0].shape;
		self->outputs[0]->format = nn->inputs[0].format;
		self->outputs[0]->data = nn->inputs[0].data;
		self->outputs[0]->data_size = nn->inputs[0].data_size;
		return 0;
	} else {
		self->outputs[0]->data = info->allocated_output;
	}

	if (nn->n_inputs != self->n_outputs) return errlog(nn,"Expected %d, got %d inputs",self->n_outputs,nn->n_inputs);

	if( nn->batchseq.graph_batches != 0 ){
		return input_execute_multibatch(self,nn);
	}
	// copy tensors using multithread overlapped copy mechanism
	int errors = 0;
	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	//
	// Do not return without waiting for mcman
	//

	for (int i = 0; i < self->n_outputs; i++) {
		struct tensor *out = self->outputs[i];
		struct tensor const *in = &nn->inputs[i];
		/* Warning! Inputs come in as max_size not data_size! */
		unsigned dsize = in->max_size;
		logmsg(nn,9,"in: [%d,%d,%d,%d] size=%d",
			in->shape.batches,
			in->shape.height,
			in->shape.width,
			in->shape.depth,
			dsize);
		if (out->max_size < dsize) {
			errlog(nn,"out %d too small (%lu < %lu)", i, out->max_size, dsize);
			out->shape.batches = 0;
			out->shape.height = 0;
			out->shape.width = 0;
			out->shape.depth = 0;
			out->data_size = 0;
			errors = 1;
		}else{
			if( dsize > 0 )
				nn_mcmanager_vmemcpy( nn, &mcman, out->data,in->data,dsize);
			out->shape = in->shape;
			out->format = in->format;
			out->data_size = dsize;
		}
	}
	nn_mcmanager_wait( nn, &mcman );

	if( errors )
		return -1;
	return 0;
}

// set up the batch slicing strategy:
//  Sets 
//    bsp->total_batches = batches
//    bsp->n_iters, n_iters_1, batch_n1, batch_n2
//
//	So that n_iters_1 * batch_n1 + (n_iters-n_iters_1)*batch_n2 = batches
//   and batch_n1, batch_n2 both <= graph_batches, 
//   and minimizing n_iters_1.
//    
static void
set_batch_slicing( struct nn_graph_batchseqstate * bsp, int batches )
{
	int graph_batches = bsp->graph_batches;
	int batch_quant = bsp->batch_quant;
	bsp->total_batches = batches;

	// determine the run strategy: we need to do 'batches' in runs of graph_batches.
	if( batches <= graph_batches ){			// <= 1 batch
		bsp->n_iters = bsp->n_iters_1 = 1;
		bsp->batch_n1 = batches;
		return;
	}
	// nruns >= 2
	int nruns = (unsigned)( batches + (graph_batches-1))/(unsigned)graph_batches;
	bsp->n_iters = bsp->n_iters_1 = nruns;
	if( nruns * graph_batches == batches ){				// exact slicing
		bsp->batch_n1 = graph_batches;				// divides into full runs.
		return;
	}
	if( (bsp->options & 1)== 0 ){
		// does the work divide into batch_quant*nruns?
		unsigned nq = batch_quant*nruns;
		unsigned per_run = batches/nq;
		if( per_run*nq == batches ){		// yes it does
			bsp->batch_n1 = per_run*batch_quant;
			return;
		}
	}
	// ok, we've mostly run out of clever ideas. We will do all but 1 @ graph_batches,
	// and one odd; or we'll split the last two in two equal, if both are multiples
	// of quant
	unsigned last2rem = batches - graph_batches*(nruns-2);
	bsp->batch_n1 = graph_batches;
	if(  last2rem % (unsigned)(2*batch_quant)== 0 ){	// split last two in 2.
		if( nruns == 2 ){
			bsp->batch_n1 = last2rem >>1;
		}else{
			bsp->n_iters_1 = nruns-2;
			bsp->batch_n2 = last2rem >>1;
		}
	}else{
		bsp->n_iters_1 = nruns-1;
		bsp->batch_n2 = last2rem-graph_batches;
	}
	// one last thing: if n_iters_1 < nruns, we have at least one 'graph_batches'
	// followed by 1 or 2 smaller. If the previous exec did *not* end in 'graph_batches',
	// switch these around (more likely to a avoid a change in size across runs)
	// This can be disabled with bit 1 of the options.
	//
	if(  bsp->n_iters_1  < nruns 
		&& (bsp->options & 2)== 0 
		&& bsp->batchn != graph_batches ){
		bsp->n_iters_1 = nruns - bsp->n_iters_1;	// 1 or 2
		bsp->batch_n1 = bsp->batch_n2;
		bsp->batch_n2 = graph_batches;
	}
}

static int input_execute_multibatch(struct nn_node *self, struct nn_graph *nn)
{
	struct nn_batchseq_portdesc *inseq_desc = nn->batchseq.inseq_desc;
	int n_inputs = nn->n_inputs;
	
	if( nn->batchseq.iterno == 0 ){	// first time through
		if( inseq_desc == NULL ){		// need to allocate these
			inseq_desc = (struct nn_batchseq_portdesc *)nn_calloc( n_inputs, sizeof(struct nn_batchseq_portdesc));
			if( inseq_desc == NULL) return errlog(nn,"alloc failed");
			nn->batchseq.inseq_desc = inseq_desc;
		}
		int n_dimsel = nn->batchseq.n_dimsel_in;
		int32_t const * dimsels = nn->batchseq.dimsel_in;
		int dimsel = dimsels[0];
		int batches = -1;
		
		for( int i = 0; i < n_inputs; i++ ){
			if( i < n_dimsel) dimsel = dimsels[i];
			if( dimsel < 0 ){
				inseq_desc[i].batchsize = 0;
			}else{
				struct tensor const * in_tens = &nn->inputs[i];
				if( batches < 0 ) batches = in_tens->shape.dimension[dimsel];
				
				uint32_t product = 1;
				for( int j = 0; j < 4; j++ ){
					uint32_t d = in_tens->shape.dimension[j];
					if( j == dimsel ){
						if( d != batches ) return errlog(nn,"unexpected size on dim %d of input %d",j,i);
						if( product != 1) return errlog(nn, "unsupported slicing on input %d", i);
					}else{
						product *= d;
					}
				}
				uint32_t batchsize = product * self->output_defs[i].elementsize;
				inseq_desc[i].batchsize = batchsize;
				if( batchsize*batches != in_tens->data_size )
					return errlog(nn,"size mismatch on input %d, expected %d*%d got %d",
						i, batches, (int)batchsize, (int)in_tens->data_size);
			}
		}
		// determine the slicing strategy
		set_batch_slicing( &nn->batchseq, batches );		
		// set up the first slice
		nn->batchseq.batchn = nn->batchseq.batch_n1;
		nn->batchseq.batchoffs = 0;
	}
	// go through all the inputs and copy the data for the current batch
	unsigned batchoff = nn->batchseq.batchoffs;
	unsigned batch_n = nn->batchseq.batchn;
	logmsg(nn,1,"iteration %d: %d batches at %d within %d",
		(int)nn->batchseq.iterno, batch_n, batchoff, (int)nn->batchseq.total_batches);
		
	int errors = 0;
	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	//
	// Do not return without waiting for mcman
	//
	int n_dimsel = nn->batchseq.n_dimsel_in;
	int32_t const * dimsels = nn->batchseq.dimsel_in;
	int dimsel = dimsels[0];

	for (int i = 0; i < n_inputs; i++) {
		struct tensor *out = self->outputs[i];
		struct tensor const *in = &nn->inputs[i];
		out->shape = in->shape;
		out->format = in->format;
		
		if( i < n_dimsel) dimsel = dimsels[i];
		unsigned copysize=0;
		uint8_t const * src;
		if( dimsels < 0 ){	// direct copy
			copysize = in->max_size;
			src = (uint8_t const*)in->data;
		}else if( dimsel <=3){	// (already checked; guarding stores)
			unsigned batchsize = inseq_desc[i].batchsize;
			copysize = batchsize * batch_n;
			src = (uint8_t const*)in->data + batchsize * batchoff;
			out->shape.dimension[dimsel] = batch_n;
		}
		if( copysize > 0 ){
			if( copysize > out->max_size){
				errlog(nn,"output %d: size %d, need room for %d bytes",
					i, (int)out->max_size, (int)copysize);
				errors = 1;
				copysize = 0;
			}else{
				logmsg(nn,3,"input %d: copy %d bytes @ offset %d",i, (int)copysize,(int)(src-(uint8_t const*)in->data));
				nn_mcmanager_vmemcpy( nn, &mcman, out->data,src,copysize);
			}
		}
		out->data_size = copysize;
	}
	nn_mcmanager_wait( nn, &mcman );

	if( errors) return -1;
	return 0;
}

static int input_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking input node %p",self);
	int i;
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,0,"input: fatal: NULL output");
		}
	}
	struct input_info *info;
	if (self->opaque == NULL) {
		if ((info = nn_calloc(1,sizeof(*info))) == NULL) return -1;
	} else {
		info = self->opaque;
	}
	nn_sem_init(&info->donesem,0);
	self->opaque = info;

	if(self->outputs) { //n_ouputs > 0
		info->allocated_output = self->outputs[0]->data;
	}
	else {
		info->allocated_output = NULL;
	}

	logmsg(nn,2,"input node %p check OK",self);
	return 0;
}

static int input_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque) {
		nn_free(self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self,nn);
}


struct nn_node_ops nn_ops_for_INPUT = {
	.execute = input_execute,
	.check = input_check,
	.ctor = node_alloc_common,
	.dtor = input_dtor,
	.n_inputs = NN_IOCOUNT(0),
	.n_outputs = NN_IOCOUNT_GE(0),
};
