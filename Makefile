_TEST_TARGETS:= lint_script 

.PHONY: $(_TEST_TARGETS)


# You can change GRAPHINIT to be another graph setup file


# Trick for always running with a fixed umask
#   See https://patchwork.ozlabs.org/patch/496917/
#   Causes Make to rerun with same options if umask isn't correct.
#   NOTE: Umask MUST be 4 digits to prevent infinite recursion
#         (on string miscompare)

UMASK=0002
ifneq (,$(findstring windows,$(shell uname)))
    QUERY_UMASK = $(UMASK)
else
    QUERY_UMASK = $(shell umask)
endif

ifneq ($(QUERY_UMASK),$(UMASK)) #umask
.PHONY: _all
$(MAKECMDGOALS): _all
	@:
_all:
	@echo RERUNNING WITH ALTERED UMASK ${UMASK}
	@umask $(UMASK) && $(MAKE) --no-print-directory $(MAKECMDGOALS)

else #umask

ifeq (,$(GRAPHINIT))
 GRAPHINIT := test/graphinit_small.c
endif

ifeq (,$(V))
 include hexagon/nonfastrpc.mak
else
 include glue/defines.min

 # include the variant specific .mak file.
 ifneq (,$(findstring hexagon,$(V_TARGET)))
  include hexagon/fastrpc.mak
  ifeq (,$(HEXAGON_NO_TEST))
   include test/hexagon.mak
  endif
 else
  ifeq (,$(SNPE_TEST))
   ifeq (,$(GOOG_TEST))
    ifeq (,$(CANPHONE))
     include controller/$(V_TARGET).mak
    else
     include canphone/$(V_TARGET).mak
    endif
   else
    include googtest/$(V_TARGET).mak
   endif
  else
   include snpetest/$(V_TARGET).mak
  endif
 endif

 #always last
 include $(RULES_MIN)
endif

# string_map.c includes hash table built from ops.def
#hexagon/src/string_map.o : hexagon/src/optab_hash.i

hexagon/src/optab_hash.i: scripts/make_optab_hash.py hexagon/ops/src/optab.c interface/ops.def
	gcc -E -I hexagon/include -DUSE_OS_H2 hexagon/ops/src/optab.c | python2 $< $@

# Graphviz can create a graph of the test you're running, if debug level is high enough
#   to include 'DOT:' lines printed by dot_print_graph(), and you redirect output to test.log
%.dot: %.sim
	grep DOT $< | sed "s/.*DOT://" > $@
%.png: %.dot
	dot -Tpng $< > $@

print-%  : ; @echo $* = $($*)


endif #umask


lint_script:
	time /prj/qct/coredev/hexagon/sitelinks/arch/pkg/python3/x86_64/3.4.2/bin/pylint --extension-pkg-whitelist=numpy -E ./scripts/tensor_compare.py

test: $(_TEST_TARGETS)
	echo PASSED ALL TESTS!


