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
 * This contains the code for IterationConfig node.
 */

#include <nn_graph.h>
#include <stdlib.h>


//
// The BatchSeqConf has 3 inputs:
//
// input 0:  scalar, input int32, # of batches graph is built for
//            (or array of up to 3 int32's; see below)
// input 1:  array of int32, input batch config; one per graph input.
//           value of 0..3 indicates which dimension of the input
//           is sliced for batches. If -1, it means the input is not sliced,
//           it gets used as-is on each iteration.
//           If the length of the array is less than the # of graph inputs,
//           the last entry is reused as needed.
// input 2:  array of int32, input batch config; one per graph output.
//           value of 0..3 indicates which dimension of the output
//           is sliced for batches. If -1, it means the out is not sliced,
//           only the value from the last iteration is retained.
//           If the length of the array is less than the # of graph outputs,
//           the last entry is reused as needed.
//
// Input 0 may be up to 3 values:
//          [  graph_batches, graph_quant, options ]
//       graph_quant and options default to 1,0 if omitted
//       'graph_quant' must be >= 1 and a factor of graph_batches; the algorithm
//       will try to keep runs a multiple of graph_quant, but will also try to avoid
//       changes in the run size.
//
//      Here is how the slicing is done, on each run (this is done in the INPUT node):
//      (1) Determine the number of batches 'nb' the graph has been presented with. This number
//       must be the same in each input tensor to the graph, in the dimensions indicated by
//       the 'input graph config' (except in inputs which have a -1 in that array).
//
//      (2) determine slicing
//       (2a) If nb <= graph_batches, operation will be done in one iteration of nb; otherwise
//       (2b)  runcount = ceil( nb/graph_batches) will be the total number of runs. If
//              nb = runcount * graph_batches exactly, that is how it will be done.
//       (2c) otherwise, if 'options' bit 0 is clear, and if nb is an exact multiple of graph_quant*runcount, then all runs
//             will be nb/runcount in length.
//       (2d) otherwise we will do (nrun-2) full runs of 'graph_batches'; the remaining batches are graph_batches+remnant.
//            these will be done as:
//                - two equal runs of (graph_batches+remnant)/2, if that is a multiple of graph_quant,or
//                - one of graph_batches and one of remnant.
//      Thus it always possible to describe the breakdown as 'r1 runs of n1, followed by r2 runs of n2'
//      .. where r2 is 0,1 or 2.
//


static int batchseqconf_execute(struct nn_node *self, struct nn_graph *nn)
{
	return 0;
}

int check_is_flat_ints( struct tensor const * tp ){
	if( tp == NULL ) return 0;
	if( tp->shape.batches != 1 || tp->shape.height !=1 || tp->shape.width != 1 ) return 0;
	if( tp->data_size != tp->shape.depth * sizeof(int32_t)) return 0;
	return 1;
}

static int
setup_dimsel_array( struct nn_graph *nn, struct tensor const * tens, int is_output )
{
	char const *tag = is_output ? "output":"input";
	if( !check_is_flat_ints( tens) ) return errlog(nn,"bad shape for %s dimension sel array", tag );
	int n = tens->shape.depth;
	int32_t const * p = (int32_t const*)tens->data;
	
	int32_t *arr = nn_calloc(n, sizeof(int32_t));
	if( arr == NULL ) return errlog(nn, "alloc failed");
	int any_gez = 0;
	for( int i = 0; i < n; i++ ){
		int32_t x = p[i];
		if( x < -1 || x > 3 ){
			nn_free(arr);
			return errlog(nn,"%s dimsel[%d] invalid value: %d", tag, i, (int)x);
		}
		arr[i] = x;
		if( x >= 0 ) any_gez = 1;
	}
	if( !is_output && !any_gez ){
		nn_free(arr);
		return errlog(nn,"at least one input dimsel must be >=0");
	}
	int32_t * old_arr;
	if( is_output ){
		nn->batchseq.n_dimsel_out = n;
		old_arr = nn->batchseq.dimsel_out;
		nn->batchseq.dimsel_out = arr;
	}else{
		nn->batchseq.n_dimsel_in = n;
		old_arr = nn->batchseq.dimsel_in;
		nn->batchseq.dimsel_in = arr;
	}
	if( old_arr != NULL ) nn_free( old_arr );
	return 0;
}

//
// 'check' for batchseqconf does all the work.
//


static int batchseqconf_check(struct nn_node *self, struct nn_graph *nn)
{

	if( nn->batchseq.have_batchseqconf_yet) return errlog(nn,"may only have one BatchSeqConf");

	struct tensor const * batches_tensor= self->inputs[0];
	if( !check_is_flat_ints(batches_tensor)) return errlog( nn,"bad shape for batches tensor");
	int n = batches_tensor->shape.depth;
	int32_t const *p = (int32_t const*)batches_tensor->data;
	
	int32_t graph_batches = p[0];
	int32_t batch_quant = (n>=2)? p[1]: 1;
	int32_t options = (n>=3)? p[2]: 0;
	
	if(batch_quant <1 || graph_batches < batch_quant || (unsigned)graph_batches%(unsigned)batch_quant!=0)
		return errlog( nn, "bad graph_batches or batch_quant" );
	nn->batchseq.graph_batches= graph_batches;
	nn->batchseq.batch_quant = batch_quant;
	nn->batchseq.options = options;	
	
	if( setup_dimsel_array( nn,self->inputs[1], 0 )!=0
	   || setup_dimsel_array( nn, self->inputs[2], 1) != 0 )
	   return -1;
	   
	nn->batchseq.have_batchseqconf_yet = 1;
	return 0;
}



struct nn_node_ops nn_ops_for_BatchSeqConfig = {
	.execute = batchseqconf_execute,
	.check = batchseqconf_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(0),
};


