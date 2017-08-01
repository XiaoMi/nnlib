
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
 * This contains the code to allocate / free / size up struct tensor's
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <string.h>
#ifndef __hexagon__
#include <malloc.h>
#endif

struct tensor *tensor_alloc(const struct shape *shape, size_t data_size)
{
	struct tensor *newtensor;
	if ((newtensor = (struct tensor *)malloc(sizeof(*newtensor))) == NULL) {
		return NULL;
	}
	if (data_size) {
		if ((newtensor->data = memalign(128,data_size)) == NULL) {
			free(newtensor);
			return NULL;
		}
	} else {
		newtensor->data = NULL;
	}
	newtensor->max_size = data_size;
	newtensor->shape = *shape;
	newtensor->max_size = newtensor->data_size = data_size;
	newtensor->self = newtensor;
	return newtensor;
}

struct tensor *tensor_dup(const struct tensor *src)
{
	struct tensor *dst;
	if ((dst = (struct tensor *)tensor_alloc(&src->shape,src->data_size)) == NULL) {
		return NULL;
	}
	memcpy(dst->data,src->data,src->data_size);
	return dst;
}

void tensor_free(struct tensor *tensor)
{
	if (tensor->data) free(tensor->data);
	free(tensor);
}

