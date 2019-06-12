
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
 * This contains the code to allocate / free / size up struct tensor's
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef __hexagon__
#include <malloc.h>
#endif

struct tensor *tensor_alloc(const struct shape *shape, size_t data_size)
{
	struct tensor *newtensor;
	if ((newtensor = nn_calloc(1,sizeof(*newtensor))) == NULL) {
		return NULL;
	}
	if (data_size) {
		if ((newtensor->data = nn_memalign(128,data_size)) == NULL) {
			nn_free(newtensor);
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
	if ((dst = tensor_alloc(&src->shape,src->data_size)) == NULL) {
		return NULL;
	}
	memcpy(dst->data,src->data,src->data_size);
	return dst;
}

void tensor_free(struct tensor *tensor)
{
	if (tensor->data) nn_free(tensor->data);
	nn_free(tensor);
}

void print_tensor_to_file(struct nn_graph *nn, uint32_t id, uint32_t index, const struct tensor *t)
{
#if defined(V66)
	FILE *outfile;
	char filename[255];

	if ( !tensor_is_plain(t) && !tensor_is_d32(t)) {
		errlog(nn,"Can't determine tensor format");
		return;
	}

	int elementsize = tensor_type_size(t->format.type);

	snprintf(filename, 255, "%s_%x_%u.dat", nn->enable_tensor_print_prefix, (unsigned)id, (unsigned)index);
	if ((outfile = fopen(filename, "w")) == NULL) {
		errlog(nn,"Ooops... Couldn't open file '%s'", filename);
		return;
	} else {
		logmsg(nn,1,"INFO: Writing '%s' size=%d elements from %p (starting at %p) %dx%dx%dx%d", filename, elementsize, t->data, tensor_location(t,0,0,0,0), t->shape.batches, t->shape.height, t->shape.width, t->shape.depth);
	}

	uint8_t *start = 0;
	uint8_t *next = 0;
	uint8_t *last = 0;
	int len = 0;
	int total_len = 0;
	for (int b=0; b < t->shape.batches; b++) {
		for (int h=0; h < t->shape.height; h++) {
			for (int w=0; w < t->shape.width; w++) {
				for (int d=0; d < t->shape.depth; d++) {
					// TODO - There's a faster way to do this... At least for plain tensors;
					//    fwrite(t,1,(b*h*w*d),outfile);
					for (int e=0; e<elementsize; e++) {
						if ((t->max_size > total_len) && (t->data_size > total_len)) {
							next = (uint8_t*)tensor_location(t,b,h,w,d)+e;
							if (next == last+1) {
								last = next;
								len++;
							} else {
								if (len) {
									fwrite(start, 1, len, outfile);
									total_len+=len;
								}
								len = 1;
								start = last = next;
							}
						}
					}
				}
			}
		}
	}
	if (len) {
		fwrite(start, 1, len, outfile);
		total_len+=len;
	}
	fclose(outfile);
#endif
}

#ifdef SHOWY_DEBUG
void print_tensor(const struct tensor *t, const char *str)
{
	FILE *outfile;
	char filename[255];

	uint64_t pcycle = nn_os_get_cycles(NULL);
	if ( !tensor_is_plain(t) && !tensor_is_d32(t)) {
		printf("Can't determine tensor format\n");
		return;
	}

	for (int b=0; b < t->shape.batches; b++) {
		for (int d=0; d < t->shape.depth; d++) {
			snprintf(filename, 255, "debug/%llu_%p_%d_%d.dat", pcycle, t, b, d);
			if ((outfile = fopen(filename, "w")) == NULL) {
				printf("Ooops... Couldn't open file %s\n", filename);
				return;
			}
                        fprintf(outfile, "%lu,%lu\n", t->shape.height, t->shape.width);
			for (int h=0; h < t->shape.height; h++) {
				for (int w=0; w < t->shape.width; w++) {
					putc(*(uint8_t*)tensor_location(t,b,h,w,d), outfile);
                                        //putc(*((char *) t->data + d + (t->shape.depth * (w + (t->shape.width * (h + (t->shape.height * b)))))), outfile);
				}
			}
			fclose(outfile);
		}
	}
}
#elif 0

void print_tensor(const struct tensor *t, const char *str)
{
	uint32_t cksum_final = data_cksum(t->data,t->data_size);
	printf(">>> tensor %s @ %p %lux%lux%lux%lu data @ %p size %lu cksum 0x%08lx\n",
		str,
		t,
		t->shape.batches,
		t->shape.height,
		t->shape.width,
		t->shape.depth,
		t->data,
		t->data_size,
		cksum_final);
}

#else

void print_tensor(const struct tensor *t, const char *str)
{
}

#endif

void print_tensors(const struct tensor *tensors, uint32_t n_tensors) {
	for (int i=0; i<n_tensors; i++) {
		print_tensor(tensors+i,"");
	}
}

uint32_t data_cksum(void *data, uint32_t bytes)
{
	uint64_t cksum_acc = 0;
	int32_t words = bytes >> 2;
	int32_t leftovers = bytes & 3;
	uint32_t *p = data;
	uint32_t cksum_final;
	uint32_t shamt;
	int i;
	if (bytes && p) {
		for (i = 0; i < words; i++) {
			cksum_acc += p[i];
		}
		if (leftovers) {
			shamt = 32-8*leftovers;
			cksum_final = p[words];
			cksum_final <<= shamt;
			cksum_final >>= shamt;
			cksum_acc += cksum_final;
		}
	}
	cksum_acc = (cksum_acc & 0x0FFFFFFFFULL) + (cksum_acc >> 32);
	cksum_acc = (cksum_acc & 0x0FFFFFFFFULL) + (cksum_acc >> 32);
	cksum_final = cksum_acc;
	return cksum_final;
}

