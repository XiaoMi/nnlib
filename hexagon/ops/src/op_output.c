
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
#include <math.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for an output node.
 */
static int output_execute_multibatch(struct nn_node *self, struct nn_graph *nn);

static int output_execute(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	struct tensor *out;
	const struct tensor *in;
	if( nn_option_get(nn,debug_skip_output))  return 0;

	logmsg(nn,2,"output execute. self=%p ",self);
	if (nn->n_outputs != self->n_inputs) return errlog(nn,"bad # outputs");

	if( nn->batchseq.graph_batches != 0 ){
		return output_execute_multibatch(self,nn);
	}
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman );

	int buffer_error =0;
	for (i = 0; i < nn->n_outputs; i++) {
		in = self->inputs[i];
		out = &nn->outputs[i];

		if( in->data_size > 0){
			if(nn->loopstack.n > 0 && nn_graph_output_expanded(nn, i)){
                uint32_t offset = nn_loopstack_get_offset(nn, i);
                uint32_t iteration = nn_loopstack_get_itercount(nn);

                if(iteration == 0){
                    out->shape = in->shape;
                    out->data_size = in->data_size;
                }
                else{
                    out->shape.batches += in->shape.batches;
                    out->data_size += in->data_size;
                }
                if (out->max_size < in->data_size + offset) {
                    errlog(nn,"output %d too small (%zu < %zu)",i, out->max_size, in->data_size + offset);
                    buffer_error = 1;
                    continue;
                }
                uint8_t* out_data = out->data;
                
				nn_mcmanager_vmemcpy( nn, &mcman, out_data+offset, in->data, in->data_size);
				
				nn_loopstack_increment_offset(nn, i, in->data_size);
			}
			else{
                out->shape = in->shape;
				out->data_size = in->data_size;
                if (out->max_size < in->data_size) {
                    errlog(nn,"output %d too small (%zu < %zu)",i, out->max_size, in->data_size);
					buffer_error = 1;
					continue;
                }
				nn_mcmanager_vmemcpy( nn, &mcman, out->data, in->data, in->data_size);
			}
		}
	}
	nn_mcmanager_wait( nn, &mcman);

	if (buffer_error){
		return NN_EXECUTE_BUFFER_SIZE_ERROR;
	}
	/* Copy input tensor to output */
	logmsg(nn,2,"copied %d tensors",self->n_inputs);
	return 0;
}
// multibatch output.
static int output_execute_multibatch(struct nn_node *self, struct nn_graph *nn)
{
	int iterno = nn->batchseq.iterno;
	int n_out = nn->n_outputs;
	struct nn_batchseq_portdesc *outseq_desc = nn->batchseq.outseq_desc;
	int total_batches = nn->batchseq.total_batches;
	int iter_batches = nn->batchseq.batchn;
	
	if( n_out < 1) return errlog(nn,"batch seq requires >=1 output!");

	if( iterno == 0 ){	// first time through: find all the batch sizes.
		if( outseq_desc == NULL ){		// need to allocate these
			outseq_desc = (struct nn_batchseq_portdesc *)nn_calloc( n_out, sizeof(struct nn_batchseq_portdesc));
			if( outseq_desc == NULL) return errlog(nn,"alloc failed");
			nn->batchseq.outseq_desc = outseq_desc;
		}
		int n_dimsel = nn->batchseq.n_dimsel_out;
		int32_t const * dimsels = nn->batchseq.dimsel_out;
		int dimsel = dimsels[0];

		for( int i = 0; i < n_out; i++ ){
			if( i < n_dimsel) dimsel = dimsels[i];
			if( dimsel < 0 ){
				outseq_desc[i].batchsize = 0;
			}else if( dimsel <=3){	// (already checked; guarding stores)
				struct tensor const * in_tens = self->inputs[i];				
				uint32_t product = 1;
				for( int j = 0; j < 4; j++ ){
					uint32_t d = in_tens->shape.dimension[j];
					if( j == dimsel ){
						if( d != iter_batches ) return errlog(nn,"unexpected size on dim %d of output %d",j,i);
						if( product != 1) return errlog(nn, "unsupported slicing on output %d (p=%d @%d)", i, (int)product,j);
					}else{
						product *= d;
					}
				}
				// set up the output shape,size
				struct tensor * tout = & nn->outputs[i];
				tout->shape = in_tens->shape;
				tout->shape.dimension[dimsel] = total_batches;		// shape for whole run

				uint32_t batchsize = (uint32_t)in_tens->data_size / (unsigned)iter_batches;
				if( batchsize *iter_batches != in_tens->data_size){ 
					return errlog(nn,"bad output length %d, on output %d for %d batches",
						(int)in_tens->data_size, i, iter_batches);
				}
				outseq_desc[i].batchsize = batchsize;
				batchsize *= total_batches;
				if( tout->max_size < batchsize ){
					return errlog(nn, "output %d too small: %d < %d", i,
						(int)tout->max_size , (int) batchsize );
				}
				tout->data_size = batchsize;
			}
		} // for all outputs
	} // if iterno == 0
	
	// now loop through all the outputs, doing the copy operation.
	
	int errors = 0;
	int last_iter =  iterno+1 == nn->batchseq.n_iters;
	unsigned batchoffs = nn->batchseq.batchoffs;
	
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman );

	int n_dimsel = nn->batchseq.n_dimsel_out;
	int32_t const * dimsels = nn->batchseq.dimsel_out;
	int dimsel = dimsels[0];
	for( int i = 0; i < n_out; i++ ){
		struct tensor const * in_tens = self->inputs[i];				
		struct tensor * tout = & nn->outputs[i];
		if( i < n_dimsel) dimsel = dimsels[i];
		unsigned dest_offs = 0;
		unsigned copylen = 0;
		if( dimsel < 0){	// only do these on last iter
			if( last_iter ){
				copylen = in_tens->data_size;
				if( copylen > tout->max_size ){
					errlog(nn,"output %d too small: %d < %d", i,
						(int)tout->max_size, (int)copylen);
					errors = 1;
					copylen = 0;
				}
			}
		}else{
			dest_offs = outseq_desc[i].batchsize * batchoffs;
			copylen = outseq_desc[i].batchsize * iter_batches;
			if( dest_offs + copylen > tout->data_size){
				errlog(nn,"internal error: overwrite on output %, iter %d batchoff=%d",
					i, iterno, (int)batchoffs);
				errors = 1;
				copylen = 0;
			}
		}
		if( copylen > 0 ){
			nn_mcmanager_vmemcpy( nn, &mcman, (uint8_t*)tout->data + dest_offs, in_tens->data, copylen);
		}
	}
	nn_mcmanager_wait( nn, &mcman);
	if( errors) return -1;
	/* Copy input tensor to output */
	logmsg(nn,2,"copied %d tensors",self->n_inputs);
	return 0;
}


struct nn_node_ops nn_ops_for_OUTPUT = {
	.execute = output_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(0),	// could have 0 inputs.. not very useful
	.n_outputs = NN_IOCOUNT(0),
};

