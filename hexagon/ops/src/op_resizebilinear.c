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
#include <quantize.h>
#include <math.h>

#if defined(__hexagon__)
static int32_t min(int a, int32_t b) { return((a<b) ? a : b); }
#endif

float lerp(float s, float e, float t){return s+(e-s)*t;}

float blerp(float c00, float c01, float c10, float c11, float tx, float ty){
	return lerp(lerp(c00, c01, tx), lerp(c10, c11, tx), ty);
}

static int resizebilinear_execute(struct nn_node *self, struct nn_graph *nn, int elementsize)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *newdims = (const int32_t *)newdim_tensor->data;
	const int32_t newheight = newdims[0];
	const int32_t newwidth = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	const float xscale = (float)w_in/newwidth;
	const float yscale = (float)h_in/newheight;
	uint32_t close_h;
	uint32_t close_w;
	int b,h,w, d;
	float *out = (float *)out_tensor->data;
	const float *in = (const float *)in_tensor->data;
	const float *bstart;
	const float *hstart;
	const float *wstart;
	uint32_t depth_bytes = d_in * elementsize;
	uint32_t total_bytes = b_in*newheight*newwidth*depth_bytes;

	if (total_bytes > out_tensor->max_size) return errlog(nn,"out too small");
	logmsg(nn,2,"%dx%dx%dx%d --> %dx%dx%dx%d",b_in,h_in,w_in,d_in,b_in,newheight,newwidth,d_in);
	tensor_set_shape(out_tensor,b_in,newheight,newwidth,d_in);
	out_tensor->data_size = total_bytes;

	for (b = 0; b < b_in; b++) {
		bstart = in + b*h_in*w_in*d_in;
		for (h = 0; h < newheight; h++) {
			float yfloat = h*yscale;
			float yfrac = yfloat - floor(yfloat);
			float yint = yfloat - yfrac;
			close_h = h*yscale;
			hstart = bstart + close_h*w_in*d_in;
			for (w = 0; w < newwidth; w++) {
				float xfloat = w*xscale;
				float xfrac = xfloat - floor(xfloat);
				float xint = xfloat - xfrac;
				close_w = w*xscale;
				wstart = hstart + close_w*d_in;
				for (d = 0; d < d_in; d++) {
					float f00 = bstart[(int)(min(h_in-1,(yint+0))*w_in*d_in + min(w_in-1,(xint+0))*d_in)+d];
					float f01 = bstart[(int)(min(h_in-1,(yint+0))*w_in*d_in + min(w_in-1,(xint+1))*d_in)+d];
					float f10 = bstart[(int)(min(h_in-1,(yint+1))*w_in*d_in + min(w_in-1,(xint+0))*d_in)+d];
					float f11 = bstart[(int)(min(h_in-1,(yint+1))*w_in*d_in + min(w_in-1,(xint+1))*d_in)+d];
					out[d] = blerp(f00, f01, f10, f11, xfrac, yfrac);
				}
				out += d_in;
			}
		}
	}
	return 0;
}

static int resizebilinear_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	return resizebilinear_execute(self,nn,sizeof(float));
}

static int resizebilinear_qu8_execute(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1], self->inputs[2]);
	tensor_copy(self->outputs[2], self->inputs[3]);
	return resizebilinear_execute(self, nn, sizeof(uint8_t));
}
static int resizebilinear_f_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"resizebilinear node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"resizebilinear %p check OK",self);
	return 0;
}

static int resizebilinear_qu8_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"resizebilinear node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"resizebilinear %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_ResizeBilinear_f = {
	SFINIT(.execute, resizebilinear_f_execute),
	SFINIT(  .check, resizebilinear_f_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedResizeBilinear_8 = {
	SFINIT(.execute, resizebilinear_qu8_execute),
	SFINIT(  .check, resizebilinear_qu8_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};