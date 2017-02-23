
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
 * It's useful to print out the name and type of operation something is
 */
#include <stdio.h>
#include <stdlib.h>

struct debug_info {
	unsigned int id;
	const char *name;
	const char *opname;
};

static struct debug_info *info = NULL;
static unsigned int n_infos = 0;

static int sort_info(const void *va, const void *vb)
{
	const struct debug_info *a = va;
	const struct debug_info *b = vb;
	if (a->id < b->id) return -1;
	if (a->id > b->id) return 1;
	return 0;
}

void info_for_debug(unsigned int id, const char *name, const char *opname)
{
	/* realloc */
	struct debug_info *newinfo;
	if ((newinfo = realloc(info,(1+n_infos)*sizeof(*info))) == NULL) {
		printf("Oops: realloc\n");
		exit(1);
	}
	info = newinfo;
	/* add */
	info[n_infos].id = id;
	info[n_infos].name = name;
	info[n_infos].opname = opname;
	n_infos++;
	/* qsort */
	qsort(info,n_infos,sizeof(*info),sort_info);
}

static inline struct debug_info *info_id2item(unsigned int id)
{
	/* bsearch */
	struct debug_info key;
	key.id = id;
	return bsearch(&key,info,n_infos,sizeof(*info),sort_info);
}

const char *info_id2name(unsigned int id)
{
	struct debug_info *item;
	item = info_id2item(id);
	if (item) return item->name;
	else return "?";
}

const char *info_id2opname(unsigned int id)
{
	struct debug_info *item;
	item = info_id2item(id);
	if (item) return item->opname;
	else return "?";
}

