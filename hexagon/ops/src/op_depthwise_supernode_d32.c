/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

static void dwise_supernode_execute_hvx(struct nn_graph *nn, void *vinfo)
{
	dwconv2dbbb_v60_asm(
		in_pointer, // input + start_line_offset + any padding skip
		d32_weights,
		out_pointer, // out_buf_d32_asm + start_line_offset + any padding skip
		in_next_row,
		out_next_row,
		in_next_d32,
		out_next_d32,
		in_depth,
		out_width,
		num_lines,
		filt_height,
		minmax_buf,
		fixed_recip_level_size,
		filt_sum,
		stride_height,
		4,			// what is this?  Shift Amount?
		perm_ctrl);
}

static int dwise_supernode_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Padzap */
	
	/* Filters are going to be smaller than the activations */
	
	return 0;
}

static int dwise_supernode_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 12) return errlog(nn,"dwise wrong # inputs");
	if (self->n_inputs != 3) return errlog(nn,"dwise wrong # inputs");
	/* Prepare filters */
	/* Create filter sum / bias adders */
	return 0;
}

static int dwise_supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info *info = self->opaque;
	if (info != NULL) {
		//free(info->semaphores);
		free(info->biasbuf);
		free(info->node_data);
		free(info->weights);
		free(info->minmax_buf);
		free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedDepthwiseSupernode_8x8p8to8_d32 = {
	.execute = dwise_supernode_execute,
	.check = dwise_supernode_check,
	.ctor = node_alloc_common,
	.dtor = dwise_supernode_dtor,
};

