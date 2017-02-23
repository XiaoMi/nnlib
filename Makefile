# You can change this line to be another graph setup file
GRAPHINIT := test/graphinit_med.c
GRAPHINIT := /prj/dsp/qdsp6/arch/cnn/setup/inceptionv3_uint8in.c
#GRAPHINIT := test/graphinit_test.c

# You can change this line to be image data
#TESTDATA := test/zeros_299x299.c

ifeq (,$(V))

include hexagon/nonfastrpc.mak

else

include glue/defines.min

# include the variant specific .mak file.
ifneq (,$(findstring hexagon,$(V_TARGET)))
  include hexagon/fastrpc.mak
else
  include test/$(V_TARGET).mak
endif

#always last
include $(RULES_MIN)
endif
