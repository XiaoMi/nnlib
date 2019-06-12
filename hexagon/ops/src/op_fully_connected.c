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
 * This contains the code for a fully connected layer
 */
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
/*
 * For the input, since it's often just a 1D vector, let's assume it's
 *  just all the depth elements sequentially.
 * Format of weights is flexible, but what would probably work well:
 *  Columns that we want to do dot product down are numbered 0...N
 *  Rows that define different outputs are lettered a...d
 *  a0a1a2a3 b0b1b2b3 c0c1c2c3 d0d1d2d3
 *  This lets us do vector multiply reduce
 *
 * NOTE that we may want to add some extra stride values passed here as arguments
 *      Because the weights tend to be big, so we may only want to process a chunk
 *      of them over all the batches (which is usually 1 but could be > 1)
 *
 * Assumptions you can make: 
 *  out_depth is a multiple of 128 or more
 *  in_depth is a multiple of 8 or more
 *  input and weights are aligned to vector boundaries
 */
//#define __USE_ASM__

#if defined(__USE_ASM__)
void fully_connected_asm(
    const uint8_t   *input,
    const uint8_t   *weights,
    uint8_t         *output,
    int32_t         in_batches,
    int32_t         in_depth,
    int32_t         out_depth,
    int32_t         *min_max,
    int32_t         recip,
    int32_t         *suma_vals,
    const int32_t   *biasadd);
#else
void fully_connected_ref(
    const uint8_t   *input,
    const uint8_t   *weights,       // Could these be signed?
    uint8_t         *output,
    int32_t         in_batches,
    int32_t         in_depth,
    int32_t         out_depth,
//  int32_t         in_next_batch,  // comment out
    int32_t         *min_max,
    int32_t         recip,
    int32_t         *suma_vals,     // includes sum(a)*weight_offset
    const int32_t   *biasadd)       // includes sum(b)*in_offset
{
    int32_t in_d, out_d;
    int32_t i, b;
    int32_t sum;
    int32_t min = 0x7FFFFFFF;
    int32_t max = 0x80000000;
    for (b = 0; b < in_batches; b++) {
        for (out_d = 0; out_d < out_depth; out_d++) {
            sum = biasadd[out_d] + suma_vals[b];
            for (in_d = 0; in_d < in_depth; in_d += 4) {
                for (i = 0; i < 4; i++) {
                    int32_t inval = input[b*in_depth+in_d+i];
                    int32_t batchval = weights[in_d*out_depth + 4*out_d + i];
                    sum += inval * batchval;
                }
            }
            /* Collect min/max */
            if (sum < min) min = sum;
            if (sum > max) max = sum;
            /* May need some extra variable shift or something */
            sum = ((long long)sum * recip + (1<<30))>> 31;
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            output[out_d+b*out_depth] = sum;
        }
    }
    /* For vector computation, reduce to single min/max values here */
    /* Store 32 apart so you can store a whole vector */
    min_max[0] = max;
    min_max[32] = min;
}
#endif

static int fullyconnected_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor     = self->inputs[0];
    const struct tensor *weight_tensor = self->inputs[1];
    const struct tensor *suma_tensor   = self->inputs[2];
    const struct tensor *bias_tensor   = self->inputs[3];
    const struct tensor *precip_tensor = self->inputs[4];

    struct tensor *out_tensor          = self->outputs[0];
    struct tensor *out_min_tensor      = self->outputs[1];
    struct tensor *out_max_tensor      = self->outputs[2];

    const uint8_t *input = in_tensor->data;
    const uint8_t *weights = weight_tensor->data;       
    const int32_t *biasadd = bias_tensor->data;
    const int32_t *recip = precip_tensor->data;
    int32_t *suma_vals = suma_tensor->data;

    uint8_t *output = out_tensor->data;
    int32_t *min_max = (int32_t *)(((unsigned int)nn->scratch + 127)&(~127));

    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_depth   = in_tensor->shape.depth;

    int32_t out_batches = in_batches;
    int32_t out_width   = 1;
    int32_t out_height  = 1;
    int32_t out_depth   = weight_tensor->shape.depth;

    logmsg(nn,2,"full connected execute. self=%p ",self);


	if( tensor_out_prepare_normal( out_tensor,out_batches,out_height,out_width,out_depth , NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"output too small");
	}


#if defined(__USE_ASM__)
    fully_connected_asm(input,weights,output,in_batches,in_depth,out_depth,min_max,*recip,suma_vals,biasadd);
#else
    fully_connected_ref(input,weights,output,in_batches,in_depth,out_depth,min_max,*recip,suma_vals,biasadd);
#endif

    tensor_set_single_float(out_max_tensor, min_max[ 0]);
    tensor_set_single_float(out_min_tensor, min_max[32]);

#if 0
    logmsg(nn,2,"fully connected out min/max=%f/%f ",
        tensor_get_float(out_min_tensor,0),
        tensor_get_float(out_max_tensor,0));
#endif

    logmsg(nn,2,"fully connected %p done",self);

    return 0;
}

struct nn_node_ops nn_ops_for_FullyConnected_u8 = {
    .execute = fullyconnected_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(5),
    .n_outputs = NN_IOCOUNT(3),
};

#if 0
typedef void (*asmfunc_t)(
    const uint8_t   *input,
    const uint8_t   *weights,
    uint8_t         *output,
    int32_t         in_batches,
    int32_t         in_depth,
    int32_t         out_depth,
    int32_t         *min_max,
    int32_t         recip,
    int32_t         *suma_vals,
    const int32_t   *biasadd);

struct fc_info {
	struct nn_node *self;
	asmfunc_t asmfunc;
	int32_t *biasvals;
	int32_t *suma_vals;
	int32_t *minmax_buf;
};

static int fc_layer_execute_hvx(struct nn_graph *nn, struct nn_node *self)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *weight_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	const struct tensor *weight_min_tensor = self->inputs[4];
	const struct tensor *weight_max_tensor = self->inputs[5];
	const struct tensor *bias_tensor = self->inputs[6];
	const struct tensor *bias_min_tensor = self->inputs[7];
	const struct tensor *bias_max_tensor = self->inputs[8];
	const struct tensor *out_minval_tensor = self->inputs[9];
	const struct tensor *out_maxval_tensor = self->outputs[10];

	struct tensor *out_tensor;
	struct tensor *out_min_tensor;
	struct tensor *out_max_tensor;

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_depth   = in_tensor->shape.depth;
	const uint8_t *input = in_tensor->data;
	const uint8_t *weights = weight_tensor->data;       
	const uint8_t *bias = bias_tensor->data;
	int32_t out_batches = in_batches;
	int32_t out_depth = weight_tensor->shape.filt_batches;

	int32_t *minmax_buf = nn_scratch_alloc(nn,128*2);
	int32_t *suma_buf = nn_scratch_alloc(nn,in_batches*4);

	if (minmax_buf == NULL) return errlog(nn,"oops, scratch alloc minmax");
	if (minmax_buf == NULL) return errlog(nn,"oops, scratch alloc minmax");
	if (in_depth != weight_tensor->shape.filt_depth) return errlog(nn,"bad weight shape");
	if (bias_tensor->shape.depth != out_depth) return errlog(nn,"bias depth doesn't match");
	if (tensor_out_prepare_normal(out_tensor,out_batches,1,1,out_depth,NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out too small");
	}
	if (tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"out min too small");
	}
	if (tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"out max too small");
	}

	/*
 	 * Prepare the bias buffer
	 * We may want to add a different bias to each element along the depth direction.
	 * This bias comes in on the bias/bias_min/bias_max set of values
	 * We need to convert from the input format to the number space of the product
	 * Then the assembly function will add it in.
	 * Additionally, we will include any offset for negative output values 
	 * (that is, to shift a negative value up to 0) 
	 * And also additionally, we will include the GEMSUMB value at each element, which is 
	 * the input offset times the sum of the weights at a given output.
	 */

	/* EJP: Gemsumb should be moved to prepare stage / ctor? */
	for (i = 0; i < out_depth; i++) {
		int32_t baisval = bias[i];
		float bias_fval = ((biasval - bias_offset) * bias_to_prod_ratio);
		bias_fval += min_out_prod_offset;
		int32_t gemsumb_val = gemsumb(
			weights,
			in_depth,
			out_batches,
			input_offset,
			filt_offset,
			i);
		info->biasbuf[i] = fast_roundf(bias_fval) - gemsumb_val;
	}
	/*
	 * Prepare the gemsuma values 
	 * This is the sum of the input values for each element in the batch
	 */

	for (i = 0; i < in_batches; i++) {
		info->gemsuma[i] = gemsuma(
			input,
			in_depth,
			in_batches,
			input_offset,
			filt_offset,
			i);
	}


	return 0;
}

static int fc_layer_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,fc_layer_execute_hvx,self);
}



static int fc_layer_check(struct nn_node *self, struct nn_graph *nn)
{
	struct fc_info *info = self->opaque;
	const struct tensor *filt_tensor = self->inputs[1];
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t biasbuf_elements = (filt_batches+31) & (~31);
	int32_t biasbuf_size = biasbuf_elements * sizeof(int32_t);
	int32_t in_batches = in_tensor->shape.batches;
	logmsg(nn,2,"Checking fully-connected node %p",self);

	if (info == NULL) {
		if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
			return errlog(nn,"calloc");
		}
		if ((info->biasvals = nn_memalign(128,biasbuf_size)) == NULL) {
			free(info);
			return errlog(nn,"memalign");
		}
		/* FIXME: do you need to allocate storage for rearranging the weights? */
		self->opaque = info;
	}
	/* Reserve space for minmax and GEMSUMA buffer for at least 128 batches */
	nn_scratch_grow(nn,128*2+128*4);
	logmsg(nn,2,"fully-connected node %p check OK",self);
	return 0;
}

static int supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct fc_info *info = self->opaque;
	if (info) {
		free(info->biasvals);
		free(info);
		self->opaque = NULL;
	}
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedFC_8x8p8to8 = {
	.execute = fc_layer_execute_opt,
	.check = fc_layer_check,
	.ctor = node_alloc_common,
	.dtor = fc_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedFC_8x8p8to8 = {
	.execute = fc_layer_execute_ref,
	.check = fc_layer_check,
	.ctor = node_alloc_common,
	.dtor = fc_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};
#endif

