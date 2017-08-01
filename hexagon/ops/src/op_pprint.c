
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains implementations for quantized relu node
 */

#include <nn_graph.h>
#include <string.h>
#include <stdio.h>



/* This would be great for templates... */

#define PPRINT_EXEC_FUNC(NAME,TYPE,PRINTFSTR) \
static int NAME(struct nn_node *self, struct nn_graph *nn)\
{\
	const struct tensor *in_tensor = self->inputs[0];\
	uint32_t in_batches = in_tensor->shape.batches;\
	uint32_t in_height = in_tensor->shape.height;\
	uint32_t in_width = in_tensor->shape.width;\
	uint32_t in_depth = in_tensor->shape.depth;\
	uint32_t w,x,y,z;\
	TYPE *in_data = (TYPE *)in_tensor->data;\
	logmsg(nn,2,"pprinting node %p id %x",self,self->node_id); \
	logmsg(nn,1,"bhwd = %d,%d,%d,%d",\
		in_batches,in_height,in_width,in_depth); \
	for (w = 0; w < in_batches; w++)\
	 for (y = 0; y < in_height; y++)\
	  for (x = 0; x < in_width; x++)\
	   for (z = 0; z < in_depth; z++)\
		logmsg(nn,1,"[%d,%d,%d,%d]: " PRINTFSTR ,w,y,x,z,*in_data++);\
	return 0;\
}

PPRINT_EXEC_FUNC(pprint_8_execute,uint8_t,"0x%02x")
PPRINT_EXEC_FUNC(pprint_32_execute,uint32_t,"0x%08x")
PPRINT_EXEC_FUNC(pprint_f_execute,float,"%f")

static int pprint_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 1) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 0) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"pprint check OK");
	return 0;
}


struct nn_node_ops nn_ops_for_PPrint_8 = {
	SFINIT(.execute, pprint_8_execute),
	SFINIT(  .check, pprint_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_PPrint_32 = {
	SFINIT(.execute, pprint_32_execute),
	SFINIT(  .check, pprint_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_PPrint_f = {
	SFINIT(.execute, pprint_f_execute),
	SFINIT(  .check, pprint_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

