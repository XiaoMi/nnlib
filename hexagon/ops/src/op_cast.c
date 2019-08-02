
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
#include <nn_graph.h>
#include <string.h>
#include "hvx_inlines.h"
#include "quantize.h"
#include "nn_broadcast.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a cast op. Quantized data is treated as uint8 with
 * min/max ignored.
 */

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif


enum DataType {
	TENSOR_FLOAT32 = 1,
	TENSOR_INT32,
	TENSOR_QUANT8_ASYMM
};

// data structure for int32 to float conversion thread
struct i32_to_f32_tdata {
    int whoami;
    int32_t *in_data;
    float *out_data;
    int chunk_size;
    int total_size;
    nn_sem_t donesem;
};

// run state for dequantization
struct dequant_runstate {
    uint8_t const *inp;
    float * outp;
    unsigned numel;
    unsigned chunk;		// chunks to do at once; always a multiple of 128
    volatile unsigned current_pos;	// used to share across threads.
    float qstep;
    int qzero;
    nn_sem_t done_sem;
};

// data structure for float to int32 conversion thread
struct f32_to_i32_tdata {
    float *in_data;
    int32_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

// data structure for uint8 to int32 conversion thread
struct ui8_to_i32_tdata {
    uint8_t *in_data;
    int32_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

// data structure for int32 to uint8 conversion thread
struct i32_to_ui8_tdata {
    int32_t *in_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

// data structure for float to uint8 conversion thread
struct f32_to_ui8_tdata {
    float *in_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};


// int32 to float conversion thread
// each loops through a portion of the data and performs the cast
static void i32_to_f32_worker_thread(struct nn_graph *nn, void *vtd) {

    struct i32_to_f32_tdata *td = vtd;
    int whoami = td->whoami;
    int32_t *in_data = td->in_data;
    float *out_data = td->out_data;
    int chunk_size = td->chunk_size;
    int total_size = td->total_size;
    int start = whoami * chunk_size;
    int end = (whoami + 1) * chunk_size > total_size ? total_size : (whoami + 1) * chunk_size;

    for(int i = start; i < end; i++) out_data[i] = (float)in_data[i];

    nn_sem_post(&td->donesem);
}

// dequantization thread
static void dequantize_hvx_work_func(struct nn_graph *nn, void *rstpv)
{
    struct dequant_runstate *rstp = (struct dequant_runstate *)rstpv;
    uint8_t const * inp0 = rstp->inp;
    float * outp0 = rstp->outp;
    unsigned all_numel= rstp->numel;
    unsigned chunk = rstp->chunk;
    unsigned pos;
    while ( pos  = __sync_fetch_and_add(&rstp->current_pos, chunk), pos < all_numel){
        uint8_t const * inp = inp0 + pos;
        float *outp = outp0 + pos;
        unsigned numel = min_u32(chunk, all_numel-pos);
        l2fetch(inp, 128,128, (numel+127)/128u);
        hvx_do_dequantize(inp, outp, numel, rstp->qzero,rstp->qstep);
    }
    nn_sem_post(&rstp->done_sem);
}

// float to int32 conversion thread
static void cast_float_to_int32(struct nn_graph *nn, void *vtd)
{
    struct f32_to_i32_tdata *td = vtd;
    float* in_data = td->in_data;
    int32_t* out_data = td->out_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 32);

    HVX_Vector vK_allFF = Q6_V_vsplat_R(-1);
    HVX_Vector v0x80000000 = Q6_V_vsplat_R(0x80000000);
    HVX_Vector v0xFF = Q6_V_vsplat_R(0xFF);
    HVX_Vector v156 = Q6_V_vsplat_R(156);
    HVX_Vector v31 = Q6_V_vsplat_R(31);

    for (int i=0; i<num_loops; i++) {
        HVX_Vector *fvals_p = (HVX_Vector *) in_data;
        HVX_Vector fvals = *fvals_p;
        HVX_Vector *vout = (HVX_Vector *) out_data;

        HVX_Vector mant = Q6_Vw_vasl_VwR(fvals, 8);
        mant = Q6_V_vor_VV(mant,  v0x80000000);

        HVX_Vector expval = Q6_V_vand_VV(Q6_Vw_vasr_VwR(fvals, 23), v0xFF);
        HVX_Vector rshval = Q6_Vw_vsub_VwVw(v156, expval);

        rshval= Q6_Vw_vmin_VwVw(rshval, v31);

        HVX_VectorPred not_bigval;
        not_bigval = Q6_Q_vcmp_gt_VwVw(rshval,Q6_V_vzero());
        rshval = Q6_Vw_condnac_QnVwVw(not_bigval, rshval, vK_allFF);

        HVX_VectorPred overflow = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), rshval);
        mant = Q6_V_vmux_QVV(overflow, vK_allFF, mant);
        rshval = Q6_Vw_vmax_VwVw(rshval,Q6_V_vzero());

#if __HEXAGON_ARCH__  < 65
        mant = Q6_Vuw_vlsr_VuwR(mant, 1);
#else
        mant = Q6_Vuw_vavg_VuwVuw(mant,Q6_V_vzero());
#endif
        mant = Q6_Vw_vasr_VwVw(mant, rshval);
        mant = Q6_Vw_vavg_VwVw(mant, q6op_V_vand_QnV( not_bigval, mant));
        HVX_Vector signbit = Q6_Vw_vasr_VwR(fvals, 31);
        HVX_Vector tmp;
        tmp = Q6_Vw_condacc_QnVwVw(overflow, mant, signbit);
        *vout = Q6_V_vxor_VV(tmp, signbit);

        in_data += 32;   // an HVX vector has 128 bytes which is 32 32-bit values
        out_data += 32;
    }

    nn_sem_post(&td->donesem);
}

// uint8 to int32 conversion thread
static void cast_quant8_to_int(struct nn_graph *nn, void *vtd){
    struct ui8_to_i32_tdata *td = vtd;
    uint8_t* in_data = td->in_data;
    int32_t* out_data = td->out_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128);

    for (int i=0; i<num_loops; i++) {
        HVX_Vector *vin = (HVX_Vector *) in_data;          // output is 4x bigger than input so we need 4 output vectors
        HVX_Vector *vout1 = (HVX_Vector *) out_data;
        HVX_Vector *vout2 = (HVX_Vector *) (out_data + 32);
        HVX_Vector *vout3 = (HVX_Vector *) (out_data + 64);
        HVX_Vector *vout4 = (HVX_Vector *) (out_data + 96);

        HVX_VectorPair half_words;                         // 8-bit to 16-bit
        half_words = Q6_Wuh_vunpack_Vub(*vin);

        HVX_Vector half_words_lo;
        HVX_Vector half_words_hi;
        half_words_lo = Q6_V_lo_W(half_words);
        half_words_hi = Q6_V_hi_W(half_words);

        HVX_VectorPair words_lo;                           // 16-bit to 32-bit
        HVX_VectorPair words_hi;
        words_lo = Q6_Wuw_vunpack_Vuh(half_words_lo);
        words_hi = Q6_Wuw_vunpack_Vuh(half_words_hi);

        *vout1 = Q6_V_lo_W(words_lo);
        *vout2 = Q6_V_hi_W(words_lo);
        *vout3 = Q6_V_lo_W(words_hi);
        *vout4 = Q6_V_hi_W(words_hi);

        in_data += 128;
        out_data += 128;
    }
    nn_sem_post(&td->donesem);
}

// int32 to uint8 conversion thread
static void cast_int_to_quant8(struct nn_graph *nn, void *vtd){
    struct i32_to_ui8_tdata *td = vtd;
    int32_t* in_data = td->in_data;
    uint8_t* out_data = td->out_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128);

    for (int i=0; i<num_loops; i++) {
        HVX_Vector *vin1 = (HVX_Vector *) in_data;        // input is 4x bigger than output so we need 4 input vectors
        HVX_Vector *vin2 = (HVX_Vector *) (in_data + 32);
        HVX_Vector *vin3 = (HVX_Vector *) (in_data + 64);
        HVX_Vector *vin4 = (HVX_Vector *) (in_data + 96);
        HVX_Vector *vout = (HVX_Vector *) out_data;

        HVX_Vector halfwords1;                           // 32-bit to 16-bit
        HVX_Vector halfwords2;
        halfwords1 = Q6_Vh_vpack_VwVw_sat(*vin2, *vin1);
        halfwords2 = Q6_Vh_vpack_VwVw_sat(*vin4, *vin3);

        *vout = Q6_Vub_vpack_VhVh_sat(halfwords2, halfwords1);   // 16-bit to 8-bit and store in output

        in_data += 128;
        out_data += 128;
    }

    nn_sem_post(&td->donesem);
}

// float to uint8 conversion thread
// does a float to int32 then int32 to uint8 cast
static void cast_float_to_quant8(struct nn_graph *nn, void *vtd)
{
    struct f32_to_ui8_tdata *td = vtd;
    float* in_data = td->in_data;
    uint8_t* out_data = td->out_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128);

    HVX_Vector vK_allFF = Q6_V_vsplat_R(-1);
    HVX_Vector v0x80000000 = Q6_V_vsplat_R(0x80000000);
    HVX_Vector v0xFF = Q6_V_vsplat_R(0xFF);
    HVX_Vector v156 = Q6_V_vsplat_R(156);
    HVX_Vector v31 = Q6_V_vsplat_R(31);

    for (int i=0; i<num_loops; i++) {
        HVX_Vector *vout = (HVX_Vector *) out_data;
        HVX_Vector_x4 ints;

        // float to int32
        for(int j = 0; j < 4; j++){
            HVX_Vector *fvals_p = (HVX_Vector *)(in_data + j*32);
            HVX_Vector fvals = *fvals_p;

            HVX_Vector mant = Q6_Vw_vasl_VwR(fvals, 8);
            mant = Q6_V_vor_VV(mant,  v0x80000000);

            HVX_Vector expval = Q6_V_vand_VV(Q6_Vw_vasr_VwR(fvals, 23), v0xFF);
            HVX_Vector rshval = Q6_Vw_vsub_VwVw(v156, expval);

            rshval= Q6_Vw_vmin_VwVw(rshval, v31);

            HVX_VectorPred not_bigval;
            not_bigval = Q6_Q_vcmp_gt_VwVw(rshval,Q6_V_vzero());
            rshval = Q6_Vw_condnac_QnVwVw(not_bigval, rshval, vK_allFF);

            HVX_VectorPred overflow = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), rshval);
            mant = Q6_V_vmux_QVV(overflow, vK_allFF, mant);
            rshval = Q6_Vw_vmax_VwVw(rshval,Q6_V_vzero());

#if __HEXAGON_ARCH__  < 65
            mant = Q6_Vuw_vlsr_VuwR(mant, 1);
#else
            mant = Q6_Vuw_vavg_VuwVuw(mant,Q6_V_vzero());
#endif
            mant = Q6_Vw_vasr_VwVw(mant, rshval);
            mant = Q6_Vw_vavg_VwVw(mant, q6op_V_vand_QnV( not_bigval, mant));
            HVX_Vector signbit = Q6_Vw_vasr_VwR(fvals, 31);
            HVX_Vector tmp;
            tmp = Q6_Vw_condacc_QnVwVw(overflow, mant, signbit);
            // if we only address the 'ints.val[]' with fixed indices, it will be in registers.
            ints.val[0] = ints.val[1];
            ints.val[1] = ints.val[2];
            ints.val[2] = ints.val[3];
            ints.val[3] = Q6_V_vxor_VV(tmp, signbit);
        }


        HVX_Vector halfwords1;
        HVX_Vector halfwords2;
        halfwords1 = Q6_Vh_vpack_VwVw_sat(ints.val[1], ints.val[0]);
        halfwords2 = Q6_Vh_vpack_VwVw_sat(ints.val[3], ints.val[2]);

        *vout = Q6_Vub_vpack_VhVh_sat(halfwords2, halfwords1);

        in_data += 128;
        out_data += 128;
    }

    nn_sem_post(&td->donesem);
}

static int cast_execute_int32_to_float32(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    int32_t *in_data = in_tensor->data;
    float *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_FLOAT);

    if(elements < (NUM_THREADS + 1)*10)
    {
        for(int i = 0; i < elements; i++) out_data[i] = (float)in_data[i];
        return 0;
    }

    struct i32_to_f32_tdata td[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        td[i].whoami = i;
        td[i].in_data = in_data;
        td[i].out_data = out_data;
        td[i].chunk_size = elements / NUM_THREADS + 1;
        td[i].total_size = elements;
        nn_sem_init(&td[i].donesem, 0);
    }
    for(int i = 0; i < NUM_THREADS; i++){
        nn_os_work_for_vector(nn, i32_to_f32_worker_thread, &td[i]);
    }
    for (int i=0; i<NUM_THREADS; i++) {
        nn_sem_wait(&td[i].donesem);
    }

    return 0;
}

static int cast_execute_uint8_to_float32(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    uint8_t *in_data = in_tensor->data;
    float *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_FLOAT);

    // uint8 to float is equivalent to dequantize with min=0.0, max=255.0
    struct dequant_runstate rstate;
    rstate.inp = in_data;
    rstate.outp = out_data;
    rstate.numel = elements;
    rstate.qstep = 1.f;
    rstate.qzero = saturate_u8(0);
    nn_sem_init( &rstate.done_sem,0);

    unsigned nvec = (elements+127)/128u;
    unsigned chunk = 256;
    if( nvec < 512){
        chunk = (nvec < 32)?nvec : ((nvec+1)>>1);
    }
    rstate.chunk = 128*chunk;
    rstate.current_pos = 0;
    int nthreads =  (nvec >(NUM_THREADS-1)*chunk)?NUM_THREADS: (nvec + (chunk-1))/chunk;
    for( int i =0; i < nthreads;i++) {
        nn_os_work_for_vector(nn, dequantize_hvx_work_func, &rstate);
    }
    nn_sem_wait_n_times( &rstate.done_sem, nthreads);

    return 0;
}

static int cast_execute_float32_to_int32(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    float *in_data = in_tensor->data;
    int32_t *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_INT32);

    if(elements < NUM_THREADS*128){
        for(int i = 0; i < elements; i++) {
            out_data[i] = (int32_t) in_data[i];
        }
        return 0;
    }

    int chunk_size = elements / (NUM_THREADS * 128);
    chunk_size *= 128;
    struct f32_to_i32_tdata td[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        td[i].in_data = in_data + i*chunk_size;
        td[i].out_data = out_data + i*chunk_size;
        td[i].num_elements = chunk_size;
        nn_sem_init(&td[i].donesem, 0);
    }
    for(int i = 0; i < NUM_THREADS; i++){
        nn_os_work_for_vector(nn, cast_float_to_int32, &td[i]);
    }
    for (int i=0; i<NUM_THREADS; i++) {
        nn_sem_wait(&td[i].donesem);
    }
    for(int i = NUM_THREADS*chunk_size; i < elements; i++) {
        out_data[i] = (int32_t) in_data[i];
    }

    return 0;
}

static int cast_execute_uint8_to_int32(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    uint8_t *in_data = in_tensor->data;
    int32_t *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_INT32);

    if(elements < NUM_THREADS*128){
        for(int i = 0; i < elements; i++) {
            out_data[i] = (int32_t) in_data[i];
        }
        return 0;
    }

    int chunk_size = elements / (NUM_THREADS * 128);
    chunk_size *= 128;
    struct ui8_to_i32_tdata td[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        td[i].in_data = in_data + i*chunk_size;
        td[i].out_data = out_data + i*chunk_size;
        td[i].num_elements = chunk_size;
        nn_sem_init(&td[i].donesem, 0);
    }
    for(int i = 0; i < NUM_THREADS; i++){
        nn_os_work_for_vector(nn, cast_quant8_to_int, &td[i]);
    }
    for (int i=0; i<NUM_THREADS; i++) {
        nn_sem_wait(&td[i].donesem);
    }
    for(int i = NUM_THREADS*chunk_size; i < elements; i++) {
        out_data[i] = (int32_t) in_data[i];
    }

    return 0;
}

static int cast_execute_int32_to_uint8(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    int32_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_UINT8);

    if(elements < NUM_THREADS*128){
        for(int i = 0; i < elements; i++) {
            if(in_data[i] > 255){
                out_data[i] = 255;
            }
            else if(in_data[i] < 0){
                out_data[i] = 0;
            }
            else {
                out_data[i] = (uint8_t) in_data[i];
            }
        }
        return 0;
    }

    int chunk_size = elements / (NUM_THREADS * 128);
    chunk_size *= 128;
    struct i32_to_ui8_tdata td[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++){
        td[i].in_data = in_data + i*chunk_size;
        td[i].out_data = out_data + i*chunk_size;
        td[i].num_elements = chunk_size;
        nn_sem_init(&td[i].donesem, 0);
    }
    for(int i = 0; i < NUM_THREADS; i++){
        nn_os_work_for_vector(nn, cast_int_to_quant8, &td[i]);
    }
    for (int i=0; i<NUM_THREADS; i++) {
        nn_sem_wait(&td[i].donesem);
    }
    for(int i = NUM_THREADS*chunk_size; i < elements; i++) {
        if(in_data[i] > 255){
            out_data[i] = 255;
        }
        else if(in_data[i] < 0){
            out_data[i] = 0;
        }
        else {
            out_data[i] = (uint8_t) in_data[i];
        }
    }

    return 0;
}

static int cast_execute_float32_to_uint8(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    float *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    int elements = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width * in_tensor->shape.depth;

    tensor_out_prepare_normal(out_tensor, in_tensor->shape.batches, in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth, NN_TYPE_UINT8);

    int done_by_vec = 0;

    // operation is to convert float to int32 with saturation, and then to u8 modulo 256.

    if(elements >= NUM_THREADS*128){

		int chunk_size = elements / (NUM_THREADS * 128);
		chunk_size *= 128;
		struct f32_to_ui8_tdata td[NUM_THREADS];
		for(int i = 0; i < NUM_THREADS; i++){
			td[i].in_data = in_data + i*chunk_size;
			td[i].out_data = out_data + i*chunk_size;
			td[i].num_elements = chunk_size;
			nn_sem_init(&td[i].donesem, 0);
		}
		for(int i = 0; i < NUM_THREADS; i++){
			nn_os_work_for_vector(nn, cast_float_to_quant8, &td[i]);
		}
		for (int i=0; i<NUM_THREADS; i++) {
			nn_sem_wait(&td[i].donesem);
		}
		done_by_vec = NUM_THREADS*chunk_size;
    }
    for(int i = done_by_vec; i < elements; i++) {
    	float x = in_data[i];
    	x = fminf(fmaxf(x,0.0f),255.0f);
        out_data[i] = (uint8_t) (int)x;
    }

    return 0;
}



struct nn_node_ops nn_ops_for_CastInt32ToFloat32 = {
        .execute = cast_execute_int32_to_float32,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_CastUInt8ToFloat32 = {
        .execute = cast_execute_uint8_to_float32,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_CastFloat32ToInt32 = {
        .execute = cast_execute_float32_to_int32,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_CastUInt8ToInt32 = {
        .execute = cast_execute_uint8_to_int32,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_CastInt32ToUInt8 = {
        .execute = cast_execute_int32_to_uint8,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_CastFloat32ToUInt8 = {
        .execute = cast_execute_float32_to_uint8,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(1),
        .n_outputs = NN_IOCOUNT(1),
};
