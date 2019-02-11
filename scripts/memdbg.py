#!/usr/bin/python

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

