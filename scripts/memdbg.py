#!/usr/bin/python
#
# Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the
# disclaimer below) provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * Neither the name of The Linux Foundation nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
# GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
# HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys

allocations = {}

def process_line(line,lineno):
	if not line.startswith("MEMDBG"): return
	lineitems = line.split(":")
	(linetype,srcname,srcline,memtype,ptr) = tuple(lineitems)
	if memtype.startswith("ALLOC"):
		if ptr in allocations:
			(oldsrcname,oldsrcline,oldlineno) = allocations[ptr]
			print "OOPS: allocated %p twice, first @ %s:%s (logline %d) now @ %s:%s logline %s (%s)" % (ptr,oldsrcname,oldsrcline,oldlineno,srcname,srcline,lineno,memtype)
		else:
			allocations[ptr] = (srcname,srcline,lineno)
	elif memtype.startswith("FREE"):
		if ptr not in allocations:
			print "FREED %s before allocation @ %s:%s (logline %d)" % (ptr,srcname,srcline,lineno)
		else: del allocations[ptr]
	else: raise Exception("unknown type: %s" % memtype)

def main():
	lineno = 0
	with open(sys.argv[1]) as f:
		for line in f.xreadlines():
			lineno += 1
			process_line(line.strip(),lineno)
	for (ptr,(srcname,srcline,lineno)) in sorted(allocations.items(),key=lambda x: x[1][-1]):
		print "%s NOT FREED (%s:%s logline %d" % (ptr,srcname,srcline,lineno)


if __name__=="__main__":
	main()

