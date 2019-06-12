
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


static int fake_concat_execute(struct nn_node *self, struct nn_graph *nn)
{
	int n_inputs = (self->n_inputs - 1) / 3;
	float in_min = tensor_get_float(self->inputs[1+n_inputs],0);
	float in_max = tensor_get_float(self->inputs[1+2*n_inputs],0);
	tensor_set_float(self->outputs[1],0,in_min);
	tensor_set_float(self->outputs[2],0,in_max);
	logmsg(nn,2,"Skipping fake concat min=%f max=%f",in_min,in_max);
	return 0;
}

static int fake_concat_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking fake concat");
	self->opaque = NULL;
	const struct tensor *dim_tensor = self->inputs[0];
	if (tensor_get_int32(dim_tensor,0) != 3) {
		return errlog(nn,"fake concat needs dim 3, you should have checked that.");
	}
	/* Probably should check some other stuff here */
	int n_inputs = (self->n_inputs - 1) / 3;
	if( self->n_inputs != 3*n_inputs+1){
		return errlog(nn,"bad input count");
	}
	int i;
	struct nn_node *first = self->outputs[0]->data;
	int found = 0;
	uint32_t b = self->output_defs[0].max_sizes[0];
	uint32_t h = self->output_defs[0].max_sizes[1];
	uint32_t w = self->output_defs[0].max_sizes[2];
	uint32_t d = self->output_defs[0].max_sizes[3];
	uint32_t h_pad_before = 4;
	uint32_t h_pad_after = 4;
	uint32_t w_pad_before = 4;
	uint32_t w_pad_after = (-w) & 3;
	uint32_t d_pad_before = 0;
	uint32_t d_pad_after = 0;
	uint32_t d_so_far = 0;
	/* Set my data pointer to the real location */
	self->outputs[0]->data = first->outputs[0]->data;
	/* Prepare my output tensor */
	tensor_out_prepare_padded_d32(
		self->outputs[0],
		b,
		h,h_pad_before,h_pad_after,
		w,w_pad_before,w_pad_after,
		d,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8);
	if (self->outputs[0]->data != tensor_location_d32(self->outputs[0],0,-h_pad_before,-w_pad_before,0)) {
		return errlog(nn,"math is not working today");
	}
	for (i = 0; i < n_inputs; i++) {
		/* For each input... find the node... */
		struct nn_node *tmp = find_node(nn,self->input_refs[1+i].src_id);
		if (tmp == first) {
			found++;
			//continue;
		}
		uint32_t current_d = tmp->output_defs[0].max_sizes[3];
		/* Set up the output tensor... */
		tensor_out_prepare_padded_d32(
			tmp->outputs[0],
			b,
			h,h_pad_before,h_pad_after,
			w,w_pad_before,w_pad_after,
			current_d,d_pad_before,d-current_d,
			NN_TYPE_QUINT8);
		tmp->outputs[0]->format.layout = NN_LAYOUT_D32_LOVINGLY_PREPARED;
		if (tmp->outputs[0]->format.depth_pad[1] != (d-current_d)) return errlog(nn,"depth value problem");
		tmp->outputs[0]->data = tensor_location_d32(self->outputs[0],0,-h_pad_before,-w_pad_before,d_so_far);
		d_so_far += current_d;
	}
	if (found != 1) return errlog(nn,"Expected 1 first node, found: %d",found);
	tensor_out_prepare_normal(self->outputs[1],1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(self->outputs[2],1,1,1,1,NN_TYPE_FLOAT);
	
	logmsg(nn,2,"Maybe everything is set up?");
	//return errlog(nn,"FIXME: finish this check");
	return 0;
}

static int fake_concat_dtor(struct nn_node *self, struct nn_graph *nn)
{
	self->opaque = NULL;
	return node_free_common(self,nn);
}

/*
 * Since we expect to execute quickly, try to pass the buck to previous node 
 */

static int fake_concat_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	if (self->opaque == NULL) {
		/* Maybe this should just be a logmsg and return instead of returning error */
		logmsg(nn,2,"node %p: Oops, no predecessor registered.",self);
		return 0;
	}
	struct nn_node *pred = self->opaque;
	/* Now, pass the buck */
	return pred->ops->earlywork_register(pred,nn,work);
}

static int fake_concat_earlywork_note_pred(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor)
{
	if (predecessor == NULL) return errlog(nn,"Oops: NULL predecessor");
	logmsg(nn,2,"node %p: note predecessor %p",self,predecessor);
	if (predecessor->ops->earlywork_register != NULL) self->opaque = predecessor;
	else logmsg(nn,2,"predecessor %p has no early work registration function",predecessor);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedFakeConcat_8_d32 = {
	.execute = fake_concat_execute,
	.check = fake_concat_check,
	.ctor = node_alloc_common,
	.dtor = fake_concat_dtor,
	.n_inputs = NN_IOCOUNT_GE(4),	// 1+ 3*n with n >= 1
	.n_outputs = NN_IOCOUNT(3),
	.earlywork_note_pred = fake_concat_earlywork_note_pred,
	.earlywork_register = fake_concat_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};

