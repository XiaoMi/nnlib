
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for a node that can check that 
 * input 0 matches input 1
 */

#define CHECK(THING) \
	if (dut->THING != ref->THING) return errlog(nn,"check fail " #THING " %d != %d ",(int)dut->THING,(int)ref->THING);

static int check_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *dut = self->inputs[0];
	const struct tensor *ref = self->inputs[1];
	logmsg(nn,2,"check execute. self=%p ",self);
	/* Copy input tensor to output */
	CHECK(shape.batches);
	CHECK(shape.height);
	CHECK(shape.width);
	CHECK(shape.depth);
	CHECK(data_size);
	if (memcmp(dut->data,ref->data,ref->data_size) != 0) {
		return errlog(nn,"data mismatch");
	}
	logmsg(nn,2,"check node %p OK",self);
	return 0;
}

static int check_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking check node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"check: wrong # inputs");
	if (self->n_outputs != 0) return errlog(nn,"check: wrong # outputs");
	logmsg(nn,2,"check node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Check = {
	SFINIT(.execute, check_execute),
	SFINIT(  .check, check_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

