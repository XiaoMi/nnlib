# You can change this line to be another graph setup file
GRAPHINIT := test/graphinit_med.c

# You can change this line to be image data
TESTDATA := test/zeros_299x299.c

include glue/defines.min

# include the variant specific .mak file.
ifneq (,$(findstring hexagon,$(V_TARGET)))
  include hexagon/fastrpc.mak
else
  include test/$(V_TARGET).mak
endif

#always last
include $(RULES_MIN)
