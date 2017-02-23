
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
 * This contains code for logging.
 */
#include <nn_graph.h>
#include <stdio.h>
#include <stdint.h>

void logv(const char *filename, unsigned int line, struct nn_graph *nn, int level, const char *fmt, va_list ap)
{
#if 1
	char *pos;
	uint32_t bytes_left;
	uint32_t total_bytes = 0;
	int printbytes;
	pos = nn->logbuf + nn->logbuf_pos;
	bytes_left = nn->logbuf_size - nn->logbuf_pos;
	if ((printbytes = snprintf(pos,bytes_left,"%s:%d:",filename,line)) >= bytes_left) return;
	bytes_left -= printbytes;
	pos += printbytes;
	total_bytes += printbytes;
	if ((printbytes = vsnprintf(pos,bytes_left,fmt,ap)) >= bytes_left) return;
	bytes_left -= printbytes;
	pos += printbytes;
	total_bytes += printbytes;
	if ((printbytes = snprintf(pos,bytes_left,"\n")) >= bytes_left) return;
	bytes_left -= printbytes;
	pos += printbytes;
	total_bytes += printbytes;
	nn->logbuf_pos += total_bytes;
#else
	/* EJP: for now, just printf */
	printf("%s:%d:",filename,line);
	vprintf(fmt,ap);
	printf("\n");
#endif
}

