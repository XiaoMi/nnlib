NOTE v.8.0.x tools yield wrong answers on HVX neural network app due to assembler issue (v.7.2. and v.8.1. tools do not show any such issue):
There is a bug in the assembler for v.8.0.x: endloop1 is not encoded correctly in the nn_graph project file hexagon/asm_src/gemmpybbw_h.S.
Users can work around this for v.8.0.x changing that gemmpybbw_h ASM file to enable ifdef adding an extra nop packet just before the endloop1.
This issue is intended to be fixed in release v.8.0.10 tool (so tools v.8.0.1 to v.8.0.9 would exhibit the problema without the extra nop)

