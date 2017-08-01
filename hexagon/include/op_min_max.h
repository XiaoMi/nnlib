
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
#ifndef OP_MINMAX_H
#define OP_MINMAX_H 1

#define CREATE_REF_OP_MIN_MAX(NAME, OPNAME, OPERATOR) \
struct NAME##_info {\
	int a_offset;\
	int b_offset;\
	int a_mult;\
	int b_mult;\
	int shift;\
	int qzero;\
};\
\
static inline uint8_t q8##NAME##_helper(uint8_t a, uint8_t b, void *v##NAME_info)\
{\
	const struct NAME##_info *info = (const struct NAME##_info *)v##NAME_info;\
	int a_offset = info->a_offset;\
	int b_offset = info->b_offset;\
	int a_mult = info->a_mult;\
	int b_mult = info->b_mult;\
	int shift = info->shift;\
	int qzero = info->qzero;\
	int aval = (((a - a_offset) * a_mult)>>shift)+qzero;\
	int bval = (((b - b_offset) * b_mult)>>shift)+qzero;\
\
	uint8_t ret = OPERATOR(aval, bval);\
\
	/* printf("ret=%d, a_new=%d b_new=%d\n",ret,aval,bval); */\
	return ret;\
}\
\
static int NAME##_q8_execute_ref(struct nn_node *self, struct nn_graph *nn)\
{\
	struct NAME##_info info;\
	const struct tensor *a_min_tensor = self->inputs[2];\
	const struct tensor *a_max_tensor = self->inputs[3];\
	const struct tensor *b_min_tensor = self->inputs[4];\
	const struct tensor *b_max_tensor = self->inputs[5];\
	struct tensor *out_min_tensor = self->outputs[1];\
	struct tensor *out_max_tensor = self->outputs[2];\
	float a_min_float = tensor_get_float(a_min_tensor,0);\
	float a_max_float = tensor_get_float(a_max_tensor,0);\
	float b_min_float = tensor_get_float(b_min_tensor,0);\
	float b_max_float = tensor_get_float(b_max_tensor,0);\
\
	float a_level_size = (a_max_float - a_min_float)/255;\
	float b_level_size = (b_max_float - b_min_float)/255;\
\
	float out_min = fminf(0.0,fminf(a_min_float, b_min_float));\
	float out_max = fmaxf(0.0, f##OPERATOR##f(a_max_float, b_max_float));\
	float out_level_size = (out_max - out_min)/255;\
	int retval;\
\
	/*int start_time, end_time;\
	start_time =  nn_os_get_cycles(nn);*/\
	tensor_set_shape(out_min_tensor,1,1,1,1);\
	tensor_set_float(out_min_tensor,0,out_min);\
	tensor_set_shape(out_max_tensor,1,1,1,1);\
	tensor_set_float(out_max_tensor,0,out_max);\
\
	info.a_offset = quantize_uint8(0.0f,a_min_float,a_max_float);\
	info.b_offset = quantize_uint8(0.0f,b_min_float,b_max_float);\
	info.shift = 12;\
	info.a_mult = ((float)(1<<info.shift))*(a_level_size / out_level_size) + 0.5;\
	info.b_mult = ((float)(1<<info.shift))*(b_level_size / out_level_size) + 0.5;\
	info.qzero = -out_min * (255/(out_max-out_min)) + 0.5;\
\
/*	\
	printf("amin/max = [%f %f], bmin/max = [%f %f], outmin/max=[%f %f]\n",a_min_float,a_max_float,b_min_float,b_max_float,out_min,out_max);\
	printf("a_off=%d b_off=%d a_mult=%d b_mult=%d out_level_size=%f a_level_size=%f a_zero=%f b_level_size=%f b_zero=%f\n",\
		info.a_offset,\
		info.b_offset,\
		info.a_mult,\
		info.b_mult,\
		out_level_size,\
		a_level_size,\
		a_level_size * info.a_offset - a_min_float,\
		b_level_size,\
		b_level_size * info.b_offset - b_min_float);\
*/	\
\
	retval = broadcast_elementwise_execute_quint8(self,nn,q8##NAME##_helper,&info);\
	/*end_time =  nn_os_get_cycles(nn);\
	printf(#NAME " ref cycles = %d\n",end_time-start_time);*/\
\
	return retval;\
}\
\
static int NAME##_q8_check(struct nn_node *self, struct nn_graph *nn)\
{\
	logmsg(nn,2,"##NAME node %p",self);\
	if (self->n_inputs != 6) return errlog(nn,"wrong # inputs");\
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");\
	logmsg(nn,2,"##NAME %p check OK",self);\
	return 0;\
}\
\
struct nn_node_ops nn_ops_for_Quantized##OPNAME##_8_ref = {\
	NAME##_q8_execute_ref,\
	NAME##_q8_check,\
	node_alloc_common,\
	node_free_common,\
};

#define COMPUTE_QMAX_QMIN(OPERATOR, ACONST, BCONST, IN1, IN2, VACONST, VBCONST, OUT)\
	va_val = (ACONST !=0)? VACONST : IN1;\
	vb_val = (BCONST !=0)? VBCONST : IN2;\
\
	/* val1e - 0, 2, 4..., val10 - 1, 3, 5, 7... */\
	val1 = Q6_Wh_vsub_VubVub(va_val, va_offset);\
\
	/* val2e - 0, 2, 4..., val20 - 1, 3, 5, 7... */\
	val2 = Q6_Wh_vsub_VubVub(vb_val, vb_offset);\
\
	/* lo(outval0) = [0, 4,  8 ... 124] => Note: each element is 4 bytes and we are in 128 Byte mode \
	   hi(outval0) = [2, 6, 10 ... 126]\
	   lo(outval1) = [1, 5,  9 ... 125]\
	   hi(outval1) = [3, 7, 11 ... 127] */\
	outval0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(val1), va_mult);\
	outval1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(val1), va_mult);\
\
	/* lo(outval2) = [0, 4,  8 ... 124] => Note: each element is 4 bytes and we are in 128 Byte mode \
	   hi(outval2) = [2, 6, 10 ... 126]\
	   lo(outval3) = [1, 5,  9 ... 125]\
	   hi(outval3) = [3, 7, 11 ... 127] */\
	outval2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(val2), vb_mult);\
	outval3 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(val2), vb_mult);\
\
	/*  Compute (((a - a_offset) * a_mult)>>shift) and
	 * (((b - b_offset) * b_mult)>>shift) - W -> h -> b
	 */\
	vtemp0 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(outval0), Q6_V_lo_W(outval0), shift);\
	vtemp1 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(outval1), Q6_V_lo_W(outval1), shift);\
	va_val = Q6_Vub_vasr_VhVhR_sat(vtemp1,vtemp0, 0);\
	vtemp0 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(outval2), Q6_V_lo_W(outval2), shift);\
	vtemp1 = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(outval3), Q6_V_lo_W(outval3), shift);\
	vb_val = Q6_Vub_vasr_VhVhR_sat(vtemp1,vtemp0, 0);\
	va_val = Q6_Vub_vadd_VubVub_sat(va_val,vqzero);\
	vb_val = Q6_Vub_vadd_VubVub_sat(vb_val,vqzero);\
	OUT = Q6_Vub_v##OPERATOR##_VubVub(va_val,vb_val);

//#define TOOLS_NOT_7_2_12
#ifdef TOOLS_NOT_7_2_12

#define PREDICATED_STORE(QR, QV, PTR, VAL) \
	if(QR != 128) \
		Q6_vmaskedstorentq_QAV(QV, PTR, VAL); \
		/*predicated_store(QV, PTR, VAL); */\
	else \
		*PTR = VAL;

 static inline void q##NAME##_hvx(\
 		uint8_t *a,\
 		uint8_t *b,\
 		uint8_t *out,\
 		void *v##NAME_info,\
 		int32_t elem,\
 		int32_t a_const_value,\
 		int32_t b_const_value){\
  \
 	const struct NAME##_info *info = v##NAME_info;\
 	HVX_Vector va_offset, vb_offset;\
 	HVX_Vector va_mult, vb_mult;\
 	HVX_Vector vqzero;\
 	HVX_Vector va_val, vb_val, vout_val, vaconst_val, vbconst_val;\
 	HVX_Vector vtemp0, vtemp1;\
 	HVX_VectorPair val1, val2;\
 	HVX_VectorPair outval0, outval1, outval2, outval3;\
 	HVX_Vector *ptr_a = (HVX_Vector *)a;\
 	HVX_Vector *ptr_b = (HVX_Vector *)b;\
 	HVX_Vector *ptr_out = (HVX_Vector *)out;\
 	int a_offset = info->a_offset;\
 	int b_offset = info->b_offset;\
 	int a_mult = info->a_mult;\
 	int b_mult = info->b_mult;\
 	int shift = info->shift;\
 	int qzero = info->qzero;\
 	int i, loopcount = elem >> 7;	/* 7 - log2(ALIGN_SIZE) */\
 	int reminder = elem % ALIGN_SIZE;\
 	int qval = (reminder % (ALIGN_SIZE/sizeof(int32_t)))<<2;\
 	HVX_VectorPred qPred = Q6_Q_vsetq_R(qval);\
 	a_const_value = (a_const_value<<8) | a_const_value;\
 	a_const_value = Q6_R_combine_RlRl(a_const_value, a_const_value);\
 	vaconst_val = Q6_V_vsplat_R(a_const_value);\
 	b_const_value = (b_const_value<<8) | b_const_value;\
 	b_const_value = Q6_R_combine_RlRl(b_const_value, b_const_value);\
 	vbconst_val = Q6_V_vsplat_R(b_const_value);\
  \
 	/* Assumption: Range of a_offset and b_offset to be with between 0 to 255 */\
 	a_offset = (a_offset<<8) | a_offset;\
 	b_offset = (b_offset<<8) | b_offset;\
 	qzero = (qzero<<8) | qzero;\
  \
 	a_offset = Q6_R_combine_RlRl(a_offset, a_offset);\
 	b_offset = Q6_R_combine_RlRl(b_offset, b_offset);\
  \
 	a_mult = Q6_R_combine_RlRl(a_mult, a_mult);\
 	b_mult = Q6_R_combine_RlRl(b_mult, b_mult);\
  \
 	va_offset = Q6_V_vsplat_R(a_offset);\
 	vb_offset = Q6_V_vsplat_R(b_offset);\
 	va_mult = Q6_V_vsplat_R(a_mult);\
 	vb_mult = Q6_V_vsplat_R(b_mult);\
 	vqzero = Q6_V_vsplat_R(qzero);\
  \
 	for(i=0;i<loopcount;i++){\
		l2fetch(ptr_a+128, 128 , 128 , 1);\
		l2fetch(ptr_b+128, 128 , 128 , 1);\
 		COMPUTE_QMAX_QMIN(OPERATOR, a_const_value, b_const_value, *ptr_a++, *ptr_b++, vaconst_val, vbconst_val, vout_val)\
 		*ptr_out++ = vout_val;\
 	}\
	l2fetch(ptr_a+128, 128 , 128 , 1);\
	l2fetch(ptr_b+128, 128 , 128 , 1);\
 	COMPUTE_QMAX_QMIN(OPERATOR, a_const_value, b_const_value, *ptr_a++, *ptr_b++, vaconst_val, vbconst_val, vout_val)\
 	PREDICATED_STORE(qval, qPred, ptr_out++, vout_val) \
  \
  }
#endif

#define CREATE_HVX_OP_MIN_MAX(NAME, OPNAME, OPERATOR) \
 static int NAME##_q8_execute_hvx(struct nn_node *self, struct nn_graph *nn) \
 { \
 	struct NAME##_info info;\
 	struct hvx_info opt_info;\
 \
 	const struct tensor *a_tensor = self->inputs[0];\
 	const struct tensor *b_tensor = self->inputs[1];\
 	const struct tensor *a_min_tensor = self->inputs[2];\
 	const struct tensor *a_max_tensor = self->inputs[3];\
 	const struct tensor *b_min_tensor = self->inputs[4];\
 	const struct tensor *b_max_tensor = self->inputs[5];\
 	struct tensor *out_tensor = self->outputs[0];\
 	struct tensor *out_min_tensor = self->outputs[1];\
 	struct tensor *out_max_tensor = self->outputs[2];\
 	int opt_flag = 0;\
 	uint8_t *a_data_pad;\
 	uint8_t *b_data_pad;\
	int elements, a_const_value, b_const_value;\
 \
	const uint8_t *a_data = (const uint8_t *)a_tensor->data;\
	const uint8_t *b_data = (const uint8_t *)b_tensor->data;\
	uint8_t *out_data = (uint8_t *)out_tensor->data;\
 \
 	/*int start_time, end_time;\
 	start_time =  nn_os_get_cycles(nn);*/\
	float a_min_float = tensor_get_float(a_min_tensor,0);\
	float a_max_float = tensor_get_float(a_max_tensor,0);\
	float b_min_float = tensor_get_float(b_min_tensor,0);\
	float b_max_float = tensor_get_float(b_max_tensor,0);\
\
	float a_level_size = (a_max_float - a_min_float)/255;\
	float b_level_size = (b_max_float - b_min_float)/255;\
\
	float out_min = fminf(0.0,fminf(a_min_float, b_min_float));\
	float out_max = fmaxf(0.0, f##OPERATOR##f(a_max_float, b_max_float));\
	float out_level_size = (out_max - out_min)/255;\
	int retval;\
\
	tensor_set_shape(out_min_tensor,1,1,1,1);\
	tensor_set_float(out_min_tensor,0,out_min);\
	tensor_set_shape(out_max_tensor,1,1,1,1);\
	tensor_set_float(out_max_tensor,0,out_max);\
\
	info.a_offset = quantize_uint8(0.0f,a_min_float,a_max_float);\
	info.b_offset = quantize_uint8(0.0f,b_min_float,b_max_float);\
	info.shift = 12;\
	info.a_mult = ((float)(1<<info.shift))*(a_level_size / out_level_size) + 0.5;\
	info.b_mult = ((float)(1<<info.shift))*(b_level_size / out_level_size) + 0.5;\
	info.qzero = -out_min * (255/(out_max-out_min)) + 0.5;\
\
	/*\
	printf("amin/max = [%f %f], bmin/max = [%f %f], outmin/max=[%f %f]\n",a_min_float,a_max_float,b_min_float,b_max_float,out_min,out_max);\
	printf("a_off=%d b_off=%d a_mult=%d b_mult=%d out_level_size=%f a_level_size=%f a_zero=%f b_level_size=%f b_zero=%f\n",\
		info.a_offset,\
		info.b_offset,\
		info.a_mult,\
		info.b_mult,\
		out_level_size,\
		a_level_size,\
		a_level_size * info.a_offset - a_min_float,\
		b_level_size,\
		b_level_size * info.b_offset - b_min_float);\
	*/\
 	/* Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data */\
 	opt_flag = check_prepare_hvx_opt(nn, a_tensor, b_tensor, out_tensor, a_data, b_data, &opt_info);\
 	a_data_pad = opt_info.a_data_pad;\
 	b_data_pad = opt_info.b_data_pad;\
 	elements = opt_info.elements;\
	a_const_value = opt_info.a_const_value;\
	b_const_value = opt_info.b_const_value;\
 \
 	if(opt_flag == 1) {\
 \
 		/* Intrinsic version of q##NAME */\
 		/*t4 =  nn_os_get_cycles(nn);*/\
		l2fetch(a_data_pad, 128 , 128 , 1);\
		l2fetch(b_data_pad, 128 , 128 , 1);\
		/*q##NAME##_hvx(a_data_pad, b_data_pad, out_data, &info, elements, a_const_value, b_const_value);*/\
		q##NAME##_asm(a_data_pad, b_data_pad, out_data, &info, elements, a_const_value, b_const_value);\
 \
 /*\
 		int i;\
 		for(i=0;i<elements;i++){\
 			out_data_pad[i] = OPERATOR( (((a_data_pad[i] - info.a_offset) * info.a_mult)>>info.shift)+info.qzero, (((b_data_pad[i] - info.b_offset) * info.b_mult)>>info.shift)+info.qzero );\
 			printf("a_data=%d, b_data=%d\n",a_data_pad[i], a_data_pad[i]);\
 			printf("out_data=0x%8x %d C out=%ld\n",out_data[i],out_data[i],out_data_pad[i]);\
 		}\
 */\
		retval = 0;\
 	}\
 	else {\
 		retval = broadcast_elementwise_execute_quint8(self,nn,q8##NAME##_helper,&info);\
	}\
	/*end_time =  nn_os_get_cycles(nn);\
	printf(#NAME " hvx cycles = %d opt_flag=%d\n", end_time-start_time, opt_flag);*/ \
	return retval;\
 }\
 \
 struct nn_node_ops nn_ops_for_Quantized##OPNAME##_8 = {\
	NAME##_q8_execute_hvx,\
	NAME##_q8_check,\
	node_alloc_common,\
	node_free_common,\
 };\

#endif

