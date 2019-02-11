#!/usr/bin/python

import png
import os
import sys


filename = sys.argv[1]

print "Filename is ", filename

dat = open(filename, 'r')

size = [int(x) for x in dat.readline().split(",")]

if size[0] < 17 or size[1] < 17:
    exit(0)
#size = [897,299]

array = []
line = dat.read(size[0])
while line:
    row = [ord(x) for x in list(line)]
    array.append(row)
    line = dat.read(size[0])

info = { 'width': size[0] }

my_png = png.from_array(array, 'L', info).save(filename+".png")


