
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
#include <math.h>
#include <quantize.h>
#include <nn_broadcast.h>
#include <nn_reduction.h>
#include <op_min_max.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */

static inline float min_helper(float a, float b, void * u)
{
	return fminf(a,b);
}

static inline float max_helper(float a, float b, void * u)
{
	return fmaxf(a,b);
}

static int min_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self,nn,fminf,INFINITY);
}

static int max_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_reduction_float(self,nn,fmaxf,-INFINITY);
}

static int minimum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_f(self,nn,min_helper,NULL);
}

static int maximum_execute(struct nn_node *self, struct nn_graph *nn)
{
	return broadcast_elementwise_execute_f(self,nn,max_helper,NULL);
}

static int minmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking min/max node %p",self);
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	if (self->inputs[0] == NULL) return errlog(nn,"NULL input 0");
	if (self->outputs[0] == NULL) return errlog(nn,"NULL output 0");
	if (self->n_inputs > 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # inputs");
	logmsg(nn,2,"min/max node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Min_f = {
	SFINIT(.execute, min_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Min_f_ref = {
	SFINIT(.execute, min_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Max_f = {
	SFINIT(.execute, max_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Max_f_ref = {
	SFINIT(.execute, max_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};


struct nn_node_ops nn_ops_for_Minimum_f = {
	SFINIT(.execute, minimum_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Maximum_f = {
	SFINIT(.execute, maximum_execute),
	SFINIT(  .check, minmax_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

#if defined(__hexagon__)
static int max(int a, int b) {return((a>b)?a:b);}
static int min(int a, int b) {return((a<b)?a:b);}
#endif

CREATE_REF_OP_MIN_MAX(maximum, Maximum, max)
CREATE_REF_OP_MIN_MAX(minimum, Minimum, min)

CREATE_HVX_OP_MIN_MAX(maximum, Maximum, max)
CREATE_HVX_OP_MIN_MAX(minimum, Minimum, min)

