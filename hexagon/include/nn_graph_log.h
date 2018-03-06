
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
#ifndef NN_GRAPH_LOG_H
#define NN_GRAPH_LOG_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for things used internally.
 */
#include <stdarg.h>

#define MAX_STRING_LEN 4096

#ifdef USE_OS_QURT
#include <HAP_farf.h>
#else
#define ALWAYS 1
#define FARF(...)
#endif


void logv(const char *filename, unsigned int line, struct nn_graph *nn, int level, const char *fmt, va_list ap);

static inline void logmsg_function(const char *filename, unsigned int line, struct nn_graph *nn, int level, const char *fmt, ...)
{
	if ((nn!=NULL) && (level > nn->debug_level)) return;
	char buffer[MAX_STRING_LEN];
	va_list ap;
	va_start(ap,fmt);
	vsnprintf(buffer,MAX_STRING_LEN,fmt,ap);
	FARF(ALWAYS,buffer);
	if (nn!=NULL) logv(filename,line,nn,level,buffer,NULL);
	va_end(ap);
}

static inline int errlog_function(const char *filename, unsigned int line, struct nn_graph *nn, const char *fmt, ...)
{
	char buffer[MAX_STRING_LEN];
	va_list ap;
	va_start(ap,fmt);
	vsnprintf(buffer,MAX_STRING_LEN,fmt,ap);
	FARF(ALWAYS,buffer);
	if (nn!=NULL) logv(filename,line,nn,0,buffer,NULL);
	va_end(ap);
	return -1;
}

// if NN_LOG_MAXLEV defined, all log(nn,lev,..) are disaabled at compile time, where lev < maxlev
// NOTE: if there are side-effects in the parameters to logmsg, they will be not
// be evaluated if the logging is disabled due to lev > NN_LOGMAX, but they will be
// evaluated if the call is disabled due to lev > nn->debug_level. So, avoid that.
//
#ifdef NN_LOG_MAXLEV
#if NN_LOG_MAXLEV < 0
// must pretend to expand the call, or you get unused variable warnings
#define logmsg(NN,LEVEL,...)  ({if(0)logmsg_function("",0,NN,LEVEL,__VA_ARGS__);})
#else
#define logmsg(NN,LEVEL,...) ({ if(( !__builtin_constant_p(LEVEL)) || (LEVEL <= NN_LOG_MAXLEV)) \
				logmsg_function(__FILE__,__LINE__,NN,LEVEL,__VA_ARGS__); })
#endif
#else // no NN_LOG_MAXLEV
#define logmsg(NN,LEVEL,...) logmsg_function(__FILE__,__LINE__,NN,LEVEL,__VA_ARGS__);
#endif


#define errlog(...) errlog_function(__FILE__,__LINE__,__VA_ARGS__)



#endif // NN_GRAPH_LOG_H
