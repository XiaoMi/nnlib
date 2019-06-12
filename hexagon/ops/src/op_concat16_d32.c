/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
//

/*
 * This contains implementations for quantized concat node
 */
#include <string.h>

#include "hvx_inlines.h"
#include <nn_asm_ops.h>
#include <nn_graph.h>
#include <quantize.h>

//#define CONCAT_REPORT_RUNTIME

// HVX intrinsic code is working (but will crash older compilers...)
// When this is removed, common alignment cases are still handled by
// hvx code in scalemem_d32.S, and weird alignment cases will be done by 'reference' code.
#ifdef HEXAGON_COMPILER_GE_8_0
#define CONCAT_HAS_HVX_INTRIN 1
#endif

// must be >=1.
// Each input is assigned to a thread; each time a thread finishes copying
// an input, it starts on the next pending input (if any).
//
#ifdef HEXAGON_V66
#define CONCAT_MAX_THREADS 4
#else
#define CONCAT_MAX_THREADS 2
#endif
//
// concat operation for d32 format
// This is coded for 16-bit d32, but it's written to allow it to support
// 8-bit d32 too -I think the overall approach here will be faster than what I did
// for the 8-bit case.
//

// Output composed by concatenating all the inputs along a given dimension
//  - input tensors must match output along all the other dims
//  - on the specified dim, the output tensor's size will be the sum of the
//    input tensor dimensions.
//
//
// To process a given input, there are two possible strategies:
// (1) aligned, unscaled
//      'unscaled' means the data values can be copied (input quantization same as output)
//      'aligned' means that the input and output are aligned in depth, so we can copy full
//          units of 32. If the concat is on a non-depth dimension, all operations are aligned;
//          if on depth dimension, it aligned if (a) the output slice starts at a depth offset
//          which is a multiple of 32 and (b) the depth of the input is a multiple of 32 (or it
//          is the last input).
//      Aiigned,unscaled operations can be done by a 4-d vmemcpy:
//             batches, height, d32, width
//      We can reverse the order of height & d32 if d32 dim is smaller than height.
//      There may still be alignment in the 'width' dimension, due to differences in left-padding
//      or (in the case of a concat on w) inconvenient slice widths) but this ie easily dealt with
//      by vmemcpy_2d.
// (2) aligned, scaled
//      Same as previous case, but the values need scaling due to differing quantization. We
//      need to use masked operations to constraint the 'edge' desitination stores to the proper byte lanes
//
// (3) unaligned, unscaled
//     This occurs in concat-on-depth when there are unconvenient depths. The operation is split
//     up so that each suboperation stores into a single output d32 slice. E.g. if we are atoring
//     depth of 40 starting at deph offset 30, the first operation writes 30 and 31 of the first slot,
//     the second writes the entire next slot, amd a third one writes 0..5 of the following slot.
//     In general, the operation to process a slot is:
//       (a) read from two adjacent input d32 slices containing the values needed;
//         combine using a vmux and a precomputed mask
//       (b) rotate (within the 32-element groups) using a precomputed vrdelta, to get the proper aligment
//       (c) Store the results, in general using a byte-mask to enable only the selected depth slots.
//     Step (c) also needs to mask w dimension, in the first and last vectors of the row.
//     In some cases it will only be necessary to read one d32 slice from the input during step (a)
//  (4) unaliugned, scaled
//     Same as (3) but scaling is done on the values, either before or after step (b) as convenient 
//      


//
// the 'strategy conatains an array of these descriptors, one for each input
//
// The section of the output to which the slice is to be written is defined by
//   - the shape here, inshape;
//   - the common 'tensor_addressing tout' with its strides and base address
//   - the 'oslice_offset' value here, to be added to tin.data to get the slice address
//   - oslice_d_before is nonzero if we are concatenating on depth, and the start depth
//     of the slice is not a multiple of 32. For depth concat, the 'oslice_offset' is always adjusted
//     to express a multiple of 32 elements, and any additional offsetting is expressed in oslice_d_before.
//
// For instance, if we are concatenating on batches or height, all of the 'oslice_offset' values
// are multiples  of tout.batch_stride or tout.height_stride; if on width, multiples of 32*element_size.
// If concatenating on depth, the oslice_offsets are all multiples of tout.d32_stride.
struct concat_strategy;
struct concat_input_desc;
typedef void (*run_function_fp)( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp);


struct concat_input_desc {
	struct tensor intens;			// the actual input tensor, as last seen.
	float in_min, in_max;			// range last seen for the input.
	struct tensor_addressing tin;	// the 'tensor addressing' for the input
	int oslice_offset;						// see above. This is in bytes
	int oslice_d_before;					// see above
	uint8_t aligned;					// true (for concat on depth) if the slice is aligned; i.e.
	                                // it starts at a depth multiple of 32 in the output (d_before=0), and either
									// its depth is a multiple of 32, or it is the last slice.
	uint8_t need_scale;				// true if scaling is needed on this input.
	int32_t scale_gain, scale_offs;	// scaling parms, if scaling is needed.
	run_function_fp run_function;	// pointer to the 'run function' to use
};
// the scaling parms are as for the 'nn_do_scaleoff_16to16' algorithm with u16->u16 conversion;
// it will do
//  out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3 + 32768]
//
// EXCEPT that for >= V62, we increase offset by 8*32768, and finish with saturation to uh
// (instead of saturation to h, followed by xor 0x8000) So then the formula is
//
//  out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3]


// strategy for the concat operation
//
struct concat_strategy {
	// the following are set up in 'check'
	uint8_t elbytes;
	uint8_t dtype;
	uint8_t has_input_ranges;
	int num_in;						// number of inputs, >= 1
	struct tensor const **data_tensors;	// pointers to the data input tensors

	uint8_t strategy_valid;
	uint8_t concat_dim;					// dimension on which to concatenate
	float out_min, out_max;				// last-used output range
	struct shape output_shape;			// the output shape for the whole operation.
	struct tensor_addressing tout;		// the 'addressing' info for the output.
	struct concat_input_desc *indescs;	// pointer to [num_in] of concat_input_desc

};

static int concat_setup_scaling( struct nn_node * self, struct nn_graph *nn, struct concat_strategy *info);
//
// analyze the input shapes
//
static int
concat_setup_strategy( struct nn_node * self, struct nn_graph *nn, struct concat_strategy *info)
{

	struct tensor const **data_tensors = info->data_tensors;
	struct tensor * output_tensor = self->outputs[0];
	int elbytes = info->elbytes;

	int num_in = info->num_in;
	int concat_dim = tensor_get_int32( self->inputs[0],0);
	// check valid
	if( concat_dim <0 || concat_dim > 3) return errlog(nn,"bad dim index");

	info->concat_dim = concat_dim;
	info->indescs[0].intens = *data_tensors[0];
	info->indescs[0].tin = nn_tensor_addressing_d32_16b( &info->indescs[0].intens);

	// (1) find accumulated shape and check the sizes

	struct shape tmpshape = info->indescs[0].intens.shape;

	uint32_t accdim = tmpshape.dimension[concat_dim];
	for( int i = 1; i < num_in; i++){
		struct concat_input_desc *indesci = &info->indescs[i];
		indesci->intens = *data_tensors[i];
		unsigned thisdim = indesci->intens.shape.dimension[concat_dim];
		tmpshape.dimension[concat_dim] = thisdim;
		if(thisdim == 0 || !shape_matches(&tmpshape, &indesci->intens.shape)){
			return errlog(nn,"input %d has incompatible dims for concat", i );
		}
		accdim += thisdim;
		indesci->tin = nn_tensor_addressing_d32_16b( &indesci->intens);
	}
	// set the overall output shape
	tmpshape.dimension[concat_dim] = accdim;
	info->output_shape = tmpshape;

	// (2) set up the output tensor shape
	// @@ scene missing
	int h_pad = 4;
	int w_pad0 = 4;
	int w_pad1 = (-(w_pad0 + info->output_shape.width))&3;
	int d_pad1 = (-info->output_shape.depth)&31;

	if( tensor_out_prepare_padded_d32( output_tensor, info->output_shape.batches,
			info->output_shape.height, h_pad, h_pad,
			info->output_shape.width, w_pad0, w_pad1,
			info->output_shape.depth, 0, d_pad1 ,info->dtype )!= 0 ){
		return errlog(nn,"failed to prepare output");
	}
	info->tout = nn_tensor_addressing_d32_16b( output_tensor);

	// now find the offsets to places where the inputs go into the output layout.

	unsigned acc_pos = 0;

	unsigned concat_stride;
	if (concat_dim != 3){
		switch(concat_dim){
		 case 0:
			concat_stride = info->tout.batch_stride;
			break;
		 case 1:
			concat_stride = info->tout.height_stride;
			break;
		 case 2:
			concat_stride = 32*elbytes;
			break;
		}
		for( int i =0; i < num_in; i++){
			struct concat_input_desc *indesci = &info->indescs[i];
			indesci->oslice_offset = acc_pos * concat_stride;
			indesci->oslice_d_before = 0;
			indesci->aligned = 1;
			indesci->need_scale = 0;
			acc_pos +=  indesci->intens.shape.dimension[concat_dim];
		}
	}else{
		concat_stride = info->tout.d32_stride;

		for( int i =0; i < num_in; i++){
			struct concat_input_desc *indesci = &info->indescs[i];
			unsigned d_rem = acc_pos%32u;
			unsigned thisdim = indesci->intens.shape.dimension[concat_dim];
			indesci->oslice_offset = (acc_pos/32u) * concat_stride;
			indesci->oslice_d_before = d_rem;
			indesci->aligned = (d_rem == 0) && (thisdim % 32u ==0);
			indesci->need_scale = 0;
			acc_pos +=  thisdim;
		}
		// last one is aligned if its oslice_d_before is zero.
		if( info->indescs[num_in-1].oslice_d_before == 0)
			info->indescs[num_in-1].aligned = 1;
	}
	// format the output range tensors
	// (using dummy values, 0.0 and 1.0)
	for( int i =1; i < 3; i++){
		if(tensor_set_single_float( self->outputs[i], (float)(i-1)) != 0){
			return errlog(nn,"output %d too small", i);
		}
	}
	info->strategy_valid = 1;
	//. deal with scaling
	return concat_setup_scaling( self,nn, info);
}

static void runfunc_aligned_unscaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp );
static void runfunc_aligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp );
static void runfunc_unaligned_unscaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp );
static void runfunc_unaligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp );

// only set up the scaling (also, picks the run function
// This is called with (tentatively) strategy_valid = 1.
// it returns <0 if an error was reported (and should clear strategy_valid in such a case)
//
static int
concat_setup_scaling( struct nn_node * self, struct nn_graph *nn, struct concat_strategy *info)
{
	if( !info->has_input_ranges){
		// just dup the scaling from input
		info->out_min = tensor_get_float(self->inputs[1],0);
		info->out_max = tensor_get_float(self->inputs[2],0);
	}else{
		int num_in = info->num_in;
		struct tensor const **data_tensors = info->data_tensors;
		// we need to find a common range
		struct tensor const **inmin_tensors = data_tensors + num_in;
		struct tensor const **inmax_tensors = inmin_tensors + num_in;
		float allmin = 0.0f;
		float allmax = 0.0f;
		for( int i =0; i < num_in; i++){
			struct concat_input_desc *indesci = &info->indescs[i];
			float inmin = tensor_get_float( inmin_tensors[i],0);
			float inmax = tensor_get_float( inmax_tensors[i],0);
			allmin= fminf(allmin, inmin);
			allmax= fmaxf(allmax, inmax);
			indesci->in_min = inmin;
			indesci->in_max = inmax;
		}
		info->out_min = allmin;
		info->out_max = allmax;
		if( info->elbytes == 2){
			float out_stepsize, out_scale;
			quantize_adjust_range_u16( &info->out_min,&info->out_max, &out_stepsize, &out_scale, allmin, allmax);
			allmin = info->out_min;
			allmax = info->out_max;
			float outzero = -allmin * out_scale;
			// identify inputs that need scaling
			for( int i =0; i < num_in; i++){
				struct concat_input_desc *indesci = &info->indescs[i];
				int is_aligned = indesci->aligned;

				if( indesci->in_min != allmin || indesci->in_max != allmax){
					indesci->need_scale = 1;
					indesci->run_function = is_aligned ? runfunc_aligned_u16scaled : runfunc_unaligned_u16scaled;
					float in_step = (indesci->in_max - indesci->in_min) * (float)(1.0/65536.0);
					float inzero = -indesci->in_min /in_step;
					// conversion gain with 18 fractional bits
					float gain = roundf(out_scale*in_step * (float)(1<<18));
					// use the rounded gain to find offset -- make sure inzero maps to outzero.
					float offs = 8.0f*outzero - (float)(1./32768.)*( gain*(inzero-32768.0f));
					indesci->scale_gain  = (int)gain;
					int scale_offs = roundf_i32( offs );
#if __HEXAGON_ARCH__ < 62
					scale_offs -= 32768*8;		// V60 ends with sat to i16, then xor 0x8000
#endif
					indesci->scale_offs = scale_offs;
				}else{
					indesci->run_function = is_aligned ? runfunc_aligned_unscaled : runfunc_unaligned_unscaled;
					indesci->need_scale = 0;
				}
			}
		}else{
			info->strategy_valid = 0;
			return errlog(nn,"need 8-bit scaling calc here");
		}
	}
	return 0;
}

// check if strategy is valid. Returns:
// -1  error found (and reported)
//  0  need complete rebuild
//  1  only redo scaling; shapes have not changed.
//  2  OK as-is.
//
// in the case of a 'Common' scale node, if shapes are the same,
//  the input range will get copied in and 2 is returned.
//
// NOTE: on a 0 return, info->strategy_valid  will be 0;
// on  a 1 or 2 return, it will be 1.
//

static int
concat_check_strategy( struct nn_node * self, struct nn_graph *nn, struct concat_strategy *info)
{
	if( !info->strategy_valid) return 0;
	if( tensor_get_int32(self->inputs[0],0)!= info->concat_dim){
		info->strategy_valid = 0;
		return 0;
	}
	// check all the input tensors are identical to what they were before.
	int num_in = info->num_in;
	struct tensor const ** data_tensors = info->data_tensors;
	struct concat_input_desc const *indescs = info->indescs;
	for( int i = 0; i  < num_in; i++ ){
		if( memcmp( &indescs[i].intens, data_tensors[i], sizeof(struct tensor))!=0){
			info->strategy_valid = 0;
			return 0;
		}
	}
	if( !info->has_input_ranges ){
		// just dup the scaling from input
		info->out_min = tensor_get_float(self->inputs[1],0);
		info->out_max = tensor_get_float(self->inputs[2],0);
		return 2;
	}

	// check if any input ranges are not the same as before
	struct tensor const **inmin_tensors = data_tensors + num_in;
	struct tensor const **inmax_tensors = inmin_tensors + num_in;
	for( int i = 0; i  < num_in; i++ ){
		if( indescs[i].in_min != tensor_get_float( inmin_tensors[i],0)
			||  indescs[i].in_max != tensor_get_float( inmax_tensors[i],0)){
			return 1;			// just need rescaling
		}
	}
	return 2;

}

struct concat_runstate {
	struct concat_strategy *info;

	volatile int curjob;
	int num_jobs;
	nn_sem_t done_sem;
};
static void concat_worker( struct nn_graph *nn, void *rstpv);


static int
concat16_execute( struct nn_node * self, struct nn_graph *nn)
{
	struct concat_strategy *info =  (struct concat_strategy *)self->opaque;
	int res = concat_check_strategy( self, nn, info);
	if( res <= 1){
		if( res == 0 )		// full rebuild
			res = concat_setup_strategy( self, nn, info);
		else if(res == 1)	// scaling only
			res = concat_setup_scaling( self, nn, info);
		if( res  < 0) return res;
	}
	// execute the strategy
	int nthreads = min_i32( CONCAT_MAX_THREADS, info->num_in);

	struct concat_runstate runstate;
	runstate.info = info;
	runstate.curjob = 0;
	runstate.num_jobs = info->num_in;
	nn_sem_init( & runstate.done_sem,0);

	for( int i = 0; i < nthreads; i++){
		nn_os_work_for_vector( nn, concat_worker, &runstate);
	}
	// output range tensors were already formatted in prepare; just store them.
	tensor_set_float( self->outputs[1], 0, info->out_min);
	tensor_set_float( self->outputs[2], 0, info->out_max);

	nn_sem_wait_n_times( &runstate.done_sem, nthreads);


	return 0;
}


static void
concat_worker( struct nn_graph *nn, void *rstpv)
{
	struct concat_runstate * rstp = (struct concat_runstate*)rstpv;
	struct concat_strategy const *info = rstp->info;

	int njobs = rstp->num_jobs;

	int ijob;

	while( ijob = __sync_fetch_and_add( &rstp->curjob, 1),  ijob < njobs){
		struct concat_input_desc const * indescp = &info->indescs[ijob];
		(*indescp->run_function)( nn, info, indescp);
	}
	nn_sem_post( &rstp->done_sem);
}

static void
runfunc_aligned_unscaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	int batches = indescp->intens.shape.batches;
	int height = indescp->intens.shape.height;
	int width = indescp->intens.shape.width;
	int nd32 = indescp->tin.nd32;

	int in_height_stride = indescp->tin.height_stride;
	int out_height_stride = info->tout.height_stride;
	int in_d32_stride = indescp->tin.d32_stride;
	int out_d32_stride = info->tout.d32_stride;

	unsigned widthbytes = width * 32 * info->elbytes;		// 2d copy width

	// if nd32 < height, reverse the height & nd32 loops
	if( nd32 < height){
		int t = nd32; nd32 = height; height = t;
		int ts = in_d32_stride; in_d32_stride = in_height_stride; in_height_stride = ts;
		ts = out_d32_stride; out_d32_stride = out_height_stride; out_height_stride = ts;
	}
	if( height <= 0) return;	// eliminate zero checks on loops.
	uint8_t const * inptr = indescp->tin.data;
	// prefetch first chunk
	l2fetch(inptr, in_d32_stride, widthbytes, nd32 );
	uint8_t *outptr = info->tout.data + indescp->oslice_offset;

	int in_batch_adj = indescp->tin.batch_stride - height*in_height_stride;
	int out_batch_adj = info->tout.batch_stride - height*out_height_stride;

	for( int b = batches; b > 0; b-- ){

		for( int h = height; h > 0; h--){
			// find the in ptr for next op, so we can prefetch it.
			uint8_t const * inptr_next = inptr + in_height_stride;
			if( h == 1){		// last h loop, go to next batch
				inptr_next += in_batch_adj;
				if( b == 1) inptr_next = NULL;
			}
			if( inptr_next != NULL)
				l2fetch(inptr_next, in_d32_stride, widthbytes, nd32 );

			vmemcpy_2d_asm(  widthbytes, nd32,			// width and height
					outptr,  out_d32_stride,		// output and stride
					inptr, in_d32_stride );
			outptr += out_height_stride;
			inptr = inptr_next;
		}
		inptr += indescp->tin.batch_stride;
		outptr += out_batch_adj;
	}
}

static inline __attribute__((unused))
uint16_t scale_u16( uint16_t inval, int scale_gain, int scale_offs)
{
	// get input-32768 as i16
	int val = (int16_t)(inval^0x8000);
	// multiply by 'gain'; >>15 with rounding; sat to i32
	int64_t prod64 = (int64_t)val * (int64_t)scale_gain;
	int32_t prod = Q6_R_sat_P(Q6_P_asrrnd_PI( prod64, 15));
	// add the offset, with saturation
	prod = Q6_R_add_RR_sat( prod, scale_offs+4);
	// >>3 with rounding, sat to i16
	int tmp = Q6_R_sath_R(prod>>3);
	return tmp ^ 0x8000;

}

// the u16 scaling, in two stages.
//
static inline HVX_Vector_x2
scale_u16_stage1( HVX_Vector vin, HVX_Vector vgain)
{
	HVX_Vector k8000 = Q6_V_vsplat_R( 0x80008000);
	HVX_Vector vin1 = Q6_V_vxor_VV( vin, k8000);
	HVX_Vector prod1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, vin1);
	HVX_Vector prod0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, Q6_Vh_vshuffe_VhVh(vin1,vin1));
	HVX_Vector_x2 res = {{ prod0, prod1 }};
	return res;
}
static inline HVX_Vector
scale_u16_stage2( HVX_Vector_x2 stg1, HVX_Vector voffs)
{
	HVX_Vector prod0 = Q6_Vw_vadd_VwVw_sat( stg1.val[0], voffs);
	HVX_Vector prod1 = Q6_Vw_vadd_VwVw_sat( stg1.val[1], voffs);
#if __HEXAGON_ARCH__ < 62
	HVX_Vector k8000 = Q6_V_vsplat_R( 0x80008000);
	return Q6_V_vxor_VV( Q6_Vh_vasr_VwVwR_rnd_sat( prod1, prod0, 3), k8000);
#else
	return Q6_Vuh_vasr_VwVwR_rnd_sat( prod1, prod0, 3);
#endif
}

// 'aligned' with scaling.
// This might need to deal with unaligned width dimension;
// and we must trim the w dimension accurately, since we could be concatenating
// in width dimension.
//

static void
runfunc_aligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	int batches = indescp->intens.shape.batches;
	int height = indescp->intens.shape.height;
	int width = indescp->intens.shape.width;
	int nd32 = indescp->tin.nd32;
	int in_height_stride = indescp->tin.height_stride;
	int out_height_stride = info->tout.height_stride;
	int in_d32_stride = indescp->tin.d32_stride;
	int out_d32_stride = info->tout.d32_stride;

	// if nd32 < height, reverse the height & nd32 loops
	if( nd32 < height){
		int t = nd32; nd32 = height; height = t;
		int ts = in_d32_stride; in_d32_stride = in_height_stride; in_height_stride = ts;
		ts = out_d32_stride; out_d32_stride = out_height_stride; out_height_stride = ts;
	}
	if( height <= 0 || nd32 <= 0) return;	// eliminate zero checks on loops.

	int elbytes = 2;
	unsigned widthbytes = width * 32 * elbytes;		// full copy width
	uint8_t const * inptr = indescp->tin.data;
	// prefetch first chunk
	l2fetch(inptr, in_d32_stride, widthbytes, nd32 );
	// account for any width misalignment. the src and dest are both width*32 elements,
	// but start and end don't need to be vector aligned, nor do they necessarily have
	// a common alignment.
	// (but they do both need to be multiples of 64, for u16 data, or 32 for u8).
	uint8_t *outptr = info->tout.data + indescp->oslice_offset;
	int dst_misalign = (size_t)outptr & 127;
	int src_misalign = (size_t)inptr & 127;
	outptr -= dst_misalign;							// align ptr
	widthbytes += dst_misalign;						// account left edge
	int widvecs_m1 = (widthbytes-1)/128u;			// total # vectors stored minus 1 (>=0)

	int vlalign_val = dst_misalign-src_misalign;	// vlalign to process with
	// do we need to do a 'preload', in order to have two vectors to make the first output?
	// if so, the input will be adjusted to point to the second vector needed, and the preload
	// will be at [-1].
	inptr -= src_misalign;
	int need_left_preload = 0;
	if( vlalign_val < 0){		// we do...
		need_left_preload = 1;
		inptr += 128;
	}
	HVX_VectorPred qn_left_mask = Q6_Q_vsetq_R(dst_misalign);
	HVX_VectorPred q_right_mask = q6op_Q_vsetq2_R(widthbytes);
	HVX_VectorPred q_zero = Q6_Q_vsetq_R(0);
	if( widvecs_m1==0){
		q_right_mask = Q6_Q_and_QQn(q_right_mask, qn_left_mask);
	}

	HVX_Vector vscale = Q6_V_vsplat_R(indescp->scale_gain);
	HVX_Vector voffs = Q6_V_vsplat_R(indescp->scale_offs);

	int in_batch_adj = indescp->tin.batch_stride - height*in_height_stride;
	int out_batch_adj = info->tout.batch_stride - height*out_height_stride;
	int out_height_adj = out_height_stride - nd32 * out_d32_stride;

	HVX_Vector vin_prev = Q6_V_vzero();
	for( int b = batches; b > 0; b-- ){
		for( int h = height; h > 0; h--){
			// find the in ptr for next op, so we can prefetch it.
			uint8_t const * inptr_next = inptr + in_height_stride;
			if( h == 1){		// last h loop, go to next batch
				inptr_next += in_batch_adj;
				if( b == 1) inptr_next = NULL;
			}
			if( inptr_next != NULL)
				l2fetch(inptr_next, in_d32_stride, widthbytes, nd32 );
			for( int id32 = 0; id32 < nd32; id32++){
				HVX_Vector const *vinp = (HVX_Vector const *)inptr;
				HVX_Vector *voutp = (HVX_Vector *)outptr;
				HVX_VectorPred qn_mask = qn_left_mask;
				if( need_left_preload ) vin_prev = vinp[-1];
				HVX_Vector vin = *vinp++;
				HVX_Vector_x2 stg1 = scale_u16_stage1( Q6_V_vlalign_VVR(vin,vin_prev,vlalign_val),vscale);

				for( int i = 0; i < widvecs_m1; i++){
					HVX_Vector vout = scale_u16_stage2( stg1, voffs);
					vin_prev = vin;
					vin = *vinp++;
					q6op_vstcc_QnAV( qn_mask, voutp, vout ); voutp ++;
					qn_mask = q_zero;
					stg1 = scale_u16_stage1( Q6_V_vlalign_VVR(vin,vin_prev,vlalign_val),vscale);
				}

				HVX_Vector vout = scale_u16_stage2( stg1, voffs);
				q6op_vstcc_QAV( q_right_mask, voutp, vout );

				inptr += in_d32_stride;
				outptr += out_d32_stride;
			}
			inptr = inptr_next;
			outptr += out_height_adj;
		}
		outptr += out_batch_adj;
	}
}

#if 0 // scalar version
static void
runfunc_aligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	int batches = indescp->intens.shape.batches;
	int height = indescp->intens.shape.height;
	int width = indescp->intens.shape.width;
	int nd32 = indescp->tin.nd32;

	int in_height_stride = indescp->tin.height_stride;
	int out_height_stride = info->tout.height_stride;
	int in_d32_stride = indescp->tin.d32_stride;
	int out_d32_stride = info->tout.d32_stride;

	unsigned width_elements = width * 32 ;		//per 'row'

	// if nd32 < height, reverse the height & nd32 loops
	if( nd32 < height){
		int t = nd32; nd32 = height; height = t;
		int ts = in_d32_stride; in_d32_stride = in_height_stride; in_height_stride = ts;
		ts = out_d32_stride; out_d32_stride = out_height_stride; out_height_stride = ts;
	}
	uint8_t *outptr = info->tout.data + indescp->oslice_offset;
	uint8_t const * inptr = indescp->tin.data;
	int scale_gain = indescp->scale_gain;
	int scale_offs = indescp->scale_offs;

	for( int b = 0; b < batches; b++ ){
		for( int h = 0; h < height; h++){
			for( int id32 = 0; id32 < nd32; id32++){
				uint16_t const * in_row_ptr = (uint16_t const*)(inptr + in_height_stride * h + in_d32_stride*id32);
				uint16_t * out_row_ptr = (uint16_t *)(outptr + out_height_stride * h + out_d32_stride*id32);

				for( int w = 0 ; w < width_elements; w++){
					out_row_ptr[w] = scale_u16( in_row_ptr[w], scale_gain, scale_offs);
				}
			}

		}
		inptr += indescp->tin.batch_stride;
		outptr += info->tout.batch_stride;
	}
}
#endif

//
// 'unaligned' operation
// This is expanded inline according to mode:
//    mode = 0 :  unscaled (u8 or qu8, according to info->elbytes)
//    mode = 1 :  u8 scaled (when implemented).
//    mode = 2 :  u16 scaled
//
//
// The 'd32' loop operates once for each *output* d32 slice,
// which could be one more slice than present in the input, e.g. if we are
// inserting a depth of 20 at offset 24, there will be 8 stored to the end of the
// first output slice and 8 at the start of the next.
// So we have:
//   - a store mask for the first output slice
//   - null mask for any 'internal' output slices
//   - a store mask for the last output slice
//  When there is only one output slice, the 'last' is merged to the first.
//
// In general, each output slice is generated by reading two adjacent input slices
//  (called d0 and d1), combining them with a vmux, and then aligning the data
//  with a vrdelta operation.
// The control for the mux and vrdelta are the same throughout all slices.
// But the first slice doesn't actually need 'd0' data; in some
//  cases the last slice doesn't need 'd1', due to truncation of the output.
//  All this can be simplified by predetermining which slices need distinct d0 and d1.
//  * On the first slice, d0 is always the first actual input, and d1 is the same.
//  * On subsequent slices, d0 is the same as the previous d1; and d1 is advanced
//    (unless we are on the last slice and it doesn't need d1).
//
//
//

static void __attribute__((always_inline))
runfunc_unaligned_TEMPLATE( struct nn_graph *nn, struct concat_strategy const *info,
		struct concat_input_desc const * indescp, int mode )
{
	int batches = indescp->intens.shape.batches;
	int height = indescp->intens.shape.height;
	int width = indescp->intens.shape.width;
	int depth = indescp->intens.shape.depth;
	int elbytes = (mode==0)?info->elbytes:mode;

	int in_height_stride = indescp->tin.height_stride;
	int out_height_stride = info->tout.height_stride;
	int in_d32_stride = indescp->tin.d32_stride;
	int out_d32_stride = info->tout.d32_stride;
	int in_batch_adjust = indescp->tin.batch_stride - height*in_height_stride;
	int out_batch_adjust = info->tout.batch_stride - height*out_height_stride;
	// account for any width misalignment. the src and dest are both width*32 elements,
	// but start and end don't need to be vector aligned, nor do they necessarily have
	// a common alignment.
	// (but they do both need to be multiples of 64, for u16 data, or 32 for u8).
	// Since this function is only used in depth concat, we don't need to mask the
	// stores on the left and right (only within depth).

	uint8_t *outptr = info->tout.data + indescp->oslice_offset;
	uint8_t const * inptr = indescp->tin.data;
	unsigned widthbytes = width * 32 * info->elbytes;		// full copy width
	int dst_misalign = (size_t)outptr & 127;
	int src_misalign = (size_t)inptr & 127;
	outptr -= dst_misalign;							// align ptr
	unsigned pfwid = widthbytes + src_misalign;
	widthbytes += dst_misalign;						// account left edge
	int widvecs_m1 = (widthbytes-1)/128u;			// total # vectors stored minus 1 (>=0)
	int vlalign_val = dst_misalign-src_misalign;	// vlalign to process with
	// do we need to do a 'preload', in order to have two vectors to make the first output?
	// if so, the input will be adjusted to point to the second vector needed, and the preload
	// will be at [-1].
	inptr -= src_misalign;
	int need_left_preload = 0;
	if( vlalign_val < 0){		// we do...
		need_left_preload = 1;
		inptr += 128;
	}
	// get the q_dep_split and the v_dep_rot, which will apply to the whole operation.
	// these are set up so that if you read values 0..63 from depth slice 0 and 1,
	//   then Q6_V_vmux_QVV( q_dep_split, vd1, vd0) will select values from d_off .. d_off+31
	//   (in each d32 lane)  and then Q6_V_vrdelta_VV( <result>, v_dep_rot) will rotate them down by doff slots
	//   (in each d32 lane), so they will be properly aligned.
	int oslice_d_before = indescp->oslice_d_before;
	int d_off = (-oslice_d_before)&31;		// offset to start reading at (0..31)
	HVX_Vector cmpreg = q6op_Vb_vsplat_R( d_off*elbytes);
	HVX_Vector modmask = q6op_Vb_vsplat_R(31*elbytes);
	HVX_Vector seqreg = Q6_V_vand_VV(  *(HVX_Vector const *)const_Count64, modmask );
	// seqreg is either { 0...31} *4,  or {0,0,2,2,4,4, ..62} * 2
	//
	HVX_VectorPred q_dep_split = Q6_Q_vcmp_gt_VubVub( cmpreg, seqreg);	// this is d_off*elbytes 1's in each segment.
	HVX_Vector v_dep_rot = Q6_V_vxor_VV( Q6_Vb_vadd_VbVb( cmpreg, seqreg), seqreg);
	// confine the rotation operation within the slices of 32 only.
	v_dep_rot = Q6_V_vand_VV( v_dep_rot,modmask );
	// dep_split must be all 1's when d_off = 0
	q_dep_split = Q6_Q_vcmp_eqor_QVbVb(q_dep_split, cmpreg, Q6_V_vzero() );

	unsigned last_d_output = oslice_d_before+depth-1;
	unsigned nd32_in = indescp->tin.nd32;
	// number of d32 slices we need to store into,minus 1
	int nd32out_m1 = last_d_output/32u;
	// now, we need masks for depth stores, for first and last d32 segment. If nd32=1, these
	// are merged.
	HVX_VectorPred qn_dep = Q6_Q_vcmp_gt_VubVub(
			q6op_Vb_vsplat_R( oslice_d_before * elbytes),seqreg);
	// keep that in 'maskset', bit 0
	HVX_Vector maskset = Q6_V_vand_QR(qn_dep, 0x01010101 );
	// right edge mask, for last depth
	qn_dep = Q6_Q_vcmp_gt_VubVub( seqreg,
				q6op_Vb_vsplat_R( (last_d_output&31) * elbytes));
	// combine to maskset, bit 1
	// if nd32_out=1, or it into the first mask too.
	maskset = Q6_V_vandor_VQR( maskset, qn_dep, nd32out_m1 >0 ? 0x02020202 : 0x03030303);

	// precompute which slices don't need d0 and d1 separately
	//  bit 0:  = 1 since first slice never needs d0
	//  bit 1:  = 1 when last slice doesn't need d1.

	int one_slice_bits  =
			((last_d_output&31)<oslice_d_before)? 3:1;

	HVX_Vector vscale,voffs;
	if( mode ==2){
		vscale = Q6_V_vsplat_R( indescp->scale_gain);
		voffs = Q6_V_vsplat_R( indescp->scale_offs);
	}
	HVX_Vector vin_prev;
/*
	printf("inserting d=%d @ %p+%d+%d   nd32_out=%d last_d_out=%d\n",
			depth, outptr,dst_misalign, oslice_d_before, nd32out_m1+1, last_d_output);
	printf("width = %d; widvecs = %d+1; vlalign =%d\n", width,widvecs_m1, vlalign_val );
*/
	for(int b = 0; b < batches; b++){
		for(int h = 0; h < height; h++){
			l2fetch( inptr - (need_left_preload? 128:0), in_d32_stride, pfwid, nd32_in);
			//
			// d32 loop
			//
			uint8_t const * inp_d1_prev = inptr;
			uint8_t * outp_w = outptr;
			for( int id32 = 0; id32 <= nd32out_m1; id32++){
				// set up the mask for stores, and the d0 and d1 inputs.
				unsigned d32_passcode = (id32==0)?0x01010101: (id32==nd32out_m1)? 0x02020202:0;
				HVX_VectorPred qn_dmask = Q6_Q_vand_VR( maskset, d32_passcode);
				uint8_t const * inp_d0 = inp_d1_prev;
				uint8_t const * inp_d1 = inp_d0;
				if( (one_slice_bits & d32_passcode)==0 )
					inp_d1 += in_d32_stride;
				inp_d1_prev = inp_d1;
				HVX_Vector const *vinp_d0 = (HVX_Vector const *)inp_d0;
				HVX_Vector const *vinp_d1 = (HVX_Vector const *)inp_d1;
				HVX_Vector * vpout = (HVX_Vector *)outp_w;
				HVX_Vector vin;
				// prepare for 'w' loop...
				// preload
				if( need_left_preload){
					vin_prev = Q6_V_vmux_QVV( q_dep_split, vinp_d1[-1], vinp_d0[-1]);
					vin_prev = Q6_V_vrdelta_VV( vin_prev, v_dep_rot);
				}
				vin = Q6_V_vmux_QVV( q_dep_split, *vinp_d1++, *vinp_d0++);
				vin = Q6_V_vrdelta_VV( vin, v_dep_rot);
				for( int i = 0; i < widvecs_m1; i++){		// could be zero iters
					HVX_Vector vin_aligned =  Q6_V_vlalign_VVR( vin, vin_prev, vlalign_val );
					HVX_Vector vout = vin_aligned;
					if( mode == 2){
						HVX_Vector_x2 stg1 = scale_u16_stage1(vin_aligned,vscale);
						vout = scale_u16_stage2( stg1, voffs);
					}
					vin_prev = vin;
					vin = Q6_V_vmux_QVV( q_dep_split, *vinp_d1++, *vinp_d0++);
					vin = Q6_V_vrdelta_VV( vin, v_dep_rot);
					q6op_vstcc_QnAV( qn_dmask, vpout, vout); vpout ++;
				}
				HVX_Vector vin_aligned = Q6_V_vlalign_VVR( vin, vin_prev, vlalign_val );
				HVX_Vector vout = vin_aligned;
				if( mode == 2){
					HVX_Vector_x2 stg1 = scale_u16_stage1(vin_aligned,vscale);
					vout = scale_u16_stage2( stg1, voffs);
				}
				q6op_vstcc_QnAV( qn_dmask, vpout, vout);
				outp_w += out_d32_stride;
			} // id32 loop

			inptr += in_height_stride;
			outptr += out_height_stride;
		}
		inptr += in_batch_adjust;
		outptr += out_batch_adjust;
	}
}


static void
runfunc_unaligned_unscaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	runfunc_unaligned_TEMPLATE( nn, info, indescp, 0 );
}
static void
runfunc_unaligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	runfunc_unaligned_TEMPLATE( nn, info, indescp, 2 );
}


#if 0
// slow scalar implementation for unaligned (scaled and unscaled)
static void
runfunc_unaligned_unscaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	int batches = indescp->intens.shape.batches;
	int height = indescp->intens.shape.height;
	int width = indescp->intens.shape.width;
	int elbytes = 2;//info->elbytes;
	int scale_gain= (1<<18);
	int scale_offs = 0;
	if( indescp->need_scale){
		scale_gain = indescp->scale_gain;
		scale_offs = indescp->scale_offs;
	}
	int32_t in_batch_stride = indescp->tin.batch_stride;
	int32_t out_batch_stride = info->tout.batch_stride;
	int32_t in_height_stride = indescp->tin.height_stride;
	int32_t out_height_stride = info->tout.height_stride;
	int32_t in_d32_stride = indescp->tin.d32_stride;
	int32_t out_d32_stride = info->tout.d32_stride;
	int32_t width_stride = 32*elbytes;

	uint8_t const * inp0 = indescp->tin.data;
	uint8_t  * outp0 = info->tout.data + indescp->oslice_offset;

	for(int b = 0; b < batches; b++){
		for(int h = 0; h < height; h++){
			for(int w = 0; w < width; w++){
				uint16_t const * inp = (uint16_t const*)(
						inp0 + b*in_batch_stride + h*in_height_stride + w*width_stride);
				uint16_t * outp = (uint16_t *)(
						outp0 + b*out_batch_stride + h*out_height_stride + w*width_stride);
				int d_wrpos = indescp->oslice_d_before;
				int d_rdpos = 0;
				int d_remain = indescp->intens.shape.depth;
				while(d_remain > 0){
					int d_amt = min_i32(d_remain, 32-max_i32(d_wrpos,d_rdpos));
					for( int i  = 0; i < d_amt; i++){
						outp[d_wrpos+i] = scale_u16( inp[d_rdpos+i], scale_gain, scale_offs);
					}
					d_remain -= d_amt;
					d_rdpos += d_amt;
					if( d_rdpos >= 32){ d_rdpos = 0; inp = (uint16_t const*)( (char const*)inp+ in_d32_stride);}
					d_wrpos += d_amt;
					if( d_wrpos >= 32){ d_wrpos = 0; outp = (uint16_t *)( (char *)outp+ out_d32_stride);}
				}
			}
		}
	}
}

static void
runfunc_unaligned_u16scaled( struct nn_graph *nn, struct concat_strategy const *info, struct concat_input_desc const * indescp )
{
	return runfunc_unaligned_unscaled(nn,info,indescp);
}
#endif

static int
concat16_check( struct nn_node * self, struct nn_graph *nn)
{
	int n_inputs = self->n_inputs;
	// figure out the number of data inputs
	int node_type = self->node_type;
	unsigned num_in;
	int has_input_ranges =0;

	if( node_type == OP_QuantizedConcat_u16_d32){
		has_input_ranges = 1;
		num_in = (n_inputs-1)/3u;
		if( (num_in*3+1)!= n_inputs){
			return errlog(nn,"concat node: can't have %d inputs", n_inputs);
		}
	}else{
		num_in = n_inputs-3;
	}

	// make the 'info' struct
	struct concat_strategy *info = (struct concat_strategy *) nn_calloc(1, sizeof(struct concat_strategy));
	if( info != NULL){
		info->indescs = (struct concat_input_desc*) nn_calloc( num_in, sizeof(struct concat_input_desc));
	}
	if( info == NULL || info->indescs == NULL){
		if( info!= NULL)nn_free(info);
		return errlog(nn, "calloc failed");
	}
	// fill in the things
	info->num_in = num_in;
	info->dtype = NN_TYPE_QUINT16;
	info->elbytes = 2;
	info->has_input_ranges = has_input_ranges;
	info->data_tensors = &self->inputs[has_input_ranges?1:3];
	self->opaque = info;
	return 0;
}

static int
concat16_dtor( struct nn_node * self, struct nn_graph *nn){
	struct concat_strategy *info = (struct concat_strategy *)self->opaque;
	if( info != NULL){
		if( info->indescs != NULL) nn_free(info->indescs);
		self->opaque = NULL;
	}
	return node_free_common(self,nn);
}

// two variants:
//  QuantizedConcat_u16_d32
//     Input 0:   scalar int, dim on which to concatenate
//     Input 1,2   output min,max
//     Input  3  ... n_in+2   - the actual data inputs
//     Input n_in+3 ... 2*n_in+2  - the 'min' values for the inputs (scalar float)
//     Input 2*n_in+3 ... 3*n_in+2  - the 'max' values for the inputs (scalar float)
//     Output 0: concatenated output
//     Output 1,2: out min,max (same as input 1,2)
//
//  This variation assumes all the inputs have the same range as specified
//   on inputs 1,2
//
//  QuantizedConcatCommon_u16_d32
//     Input 0:   scalar int, dim on which to concatenate
//     Input 1,2   output min,max
//     Input 3  ... n_in+2   - the actual data inputs
//     Output 0: concatenated output
//     Output 1,2: out min,max (same as input 1,2)

//

// inputs must be 3*k + 3,   k>=1
// IOCOUNT verifies >= 6.
struct nn_node_ops nn_ops_for_QuantizedConcat_u16_d32 = {
	.execute = concat16_execute,
	.check = concat16_check,
	.ctor = node_alloc_common,
	.dtor = concat16_dtor,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
	//.earlywork_register = concat_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

struct nn_node_ops nn_ops_for_QuantizedConcatCommon_u16_d32 = {
	.execute = concat16_execute,
	.check = concat16_check,
	.ctor = node_alloc_common,
	.dtor = concat16_dtor,
	.n_inputs = NN_IOCOUNT_GE(4),
	.n_outputs = NN_IOCOUNT(3),
	//.earlywork_register = concat_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
