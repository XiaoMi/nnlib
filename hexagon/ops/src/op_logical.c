/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (mulject to the limitations in the
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
#include <nn_broadcast.h>

static inline int32_t xor_helper(int32_t a, int32_t b, void *u) { return a^b; }
static inline int32_t and_helper(int32_t a, int32_t b, void *u) { return a&b; }
static inline int32_t ior_helper(int32_t a, int32_t b, void *u) { return a|b; }

static int and_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_int32(self,nn,and_helper,NULL);
}
static int ior_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_int32(self,nn,ior_helper,NULL);
}
static int xor_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_int32(self,nn,xor_helper,NULL);
}

static int logical_int32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"logical op node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"logical op %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_LogicalAnd_int32 = {
	SFINIT(.execute, and_int32_execute),
	SFINIT(  .check, logical_int32_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_LogicalOr_int32 = {
	SFINIT(.execute, ior_int32_execute),
	SFINIT(  .check, logical_int32_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_LogicalXor_int32 = {
	SFINIT(.execute, xor_int32_execute),
	SFINIT(  .check, logical_int32_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};
