
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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

#define PTR r0
#define AMT r1
#define RET r3:2
#define OLDVAL r4
#define NEWVAL r5
#define MASK0 r6
	.text
	.global nn_sem_add_fastpath
	.global nn_sem_add_fastpath_withret
nn_sem_add_fastpath:
nn_sem_add_fastpath_withret:
	{
		OLDVAL = memw_locked(PTR)
		MASK0 = ##0xFFFF0000
	}
	{
		p0 = bitsclr(OLDVAL,MASK0)
		if (!p0.new) jump:nt ##nn_sem_add_slowpath_withret
		NEWVAL = add(OLDVAL,AMT)
	}
	{
		memw_locked(PTR,p0) = NEWVAL
	}
	if (!p0) jump:nt nn_sem_add_fastpath
	{
		jumpr r31
		r1:0 = RET
	}
	.size nn_sem_add_fastpath_withret,.-nn_sem_add_fastpath_withret
	.size nn_sem_add_fastpath,.-nn_sem_add_fastpath

	.text
	.global nn_sem_sub_fastpath
nn_sem_sub_fastpath:
	{
		OLDVAL = memw_locked(PTR)
	}
	{
		p0 = cmph.gtu(AMT,OLDVAL)
		if (p0.new) jump:nt ##nn_sem_sub_slowpath
		NEWVAL = sub(OLDVAL,AMT)
	}
	{
		memw_locked(PTR,p0) = NEWVAL
	}
	if (!p0) jump:nt nn_sem_sub_fastpath
	jumpr r31
	.size nn_sem_sub_fastpath,.-nn_sem_sub_fastpath


	.text
	.global nn_mutex_lock_fastpath
nn_mutex_lock_fastpath:
	{
		OLDVAL = memw_locked(PTR)
	}
	{
		p0 = cmp.eq(OLDVAL,#0)
		if (!p0.new) jump:nt ##nn_mutex_lock_slowpath
		NEWVAL = #1
	}
	{
		memw_locked(PTR,p0) = NEWVAL
	}
	if (!p0) jump:nt nn_mutex_lock_fastpath
	jumpr r31
	.size nn_mutex_lock_fastpath,.-nn_mutex_lock_fastpath

	.text
	.global nn_mutex_unlock_fastpath
nn_mutex_unlock_fastpath:
	{
		OLDVAL = memw_locked(PTR)
	}
	{
		p0 = cmp.eq(OLDVAL,#1)
		if (!p0.new) jump:nt ##nn_mutex_unlock_slowpath
		NEWVAL = #0
	}
	{
		memw_locked(PTR,p0) = NEWVAL
	}
	if (!p0) jump:nt nn_mutex_unlock_fastpath
	jumpr r31
	.size nn_mutex_unlock_fastpath,.-nn_mutex_unlock_fastpath

#undef PTR
#undef AMT
#undef OLDVAL
#undef NEWVAL
#undef MASK0

#define OFFSET_HOWFULL 0
#define OFFSET_RECVIDX 4
#define OFFSET_RECVIDX_HOWFULL 0
#define OFFSET_HOWEMPTY 8
#define OFFSET_MUTEX 12
#define OFFSET_MUTEX_HOWEMPTY 8
#define OFFSET_SENDIDX 16
#define OFFSET_DATAPTR 24
#define OFFSET_ELEMENTS 28
#define OFFSET_ELEMENTS_DATAPTR 24

#define PIPE r0

#define DATAPTR r14
#define ELEMENTS r15
#define ELEMENTS_DATAPTR r15:14

#define HOWFULL r2
#define RECVIDX r3
#define RECVIDX_HOWFULL r3:2

#define NEWFULL r6
#define NEWIDX r7
#define NEWIDX_FULL r7:6

#define DATA r9:8


#define SPINS r28


	.text
	.global nn_pipe_recv_fastpath
nn_pipe_recv_fastpath:
	{
		ELEMENTS_DATAPTR = memd(PIPE+#OFFSET_ELEMENTS_DATAPTR)
		SPINS = #31
	}
.Lrecv_busy_spin:
	RECVIDX_HOWFULL = memd_locked(PIPE)
	{
		p0 = cmph.gtu(HOWFULL,#0)
		if (!p0.new) jump:nt .Lrecv_busy
		NEWIDX = add(RECVIDX,#1)
		NEWFULL = add(HOWFULL,#-1)
	}
	{
		DATA=memd(DATAPTR+RECVIDX<<#3)
		p0 = cmp.eq(NEWIDX,ELEMENTS)
		if (p0.new) NEWIDX = #0
	}
	memd_locked(PIPE,p0) = NEWIDX_FULL
	if (!p0) jump:nt .Lrecv_busy_spin // not atomic
	// Data received.  Now we need to increment howempty.
	{
		r1 = #1
		r0 = add(PIPE,#OFFSET_HOWEMPTY)
		r3:2 = DATA
		jump nn_sem_add_fastpath_withret
	}
.Lrecv_busy:
	pause(#10)
	{
		SPINS = add(SPINS,#-1)
		p0 = cmp.eq(SPINS,#0)
		if (!p0.new) jump:t .Lrecv_busy_spin
	}
	{
		jump nn_pipe_recv_slowpath
	}

#undef RECVIDX_HOWFULL 
#undef RECVIDX
#undef HOWFULL

#undef NEWIDX_FULL
#undef NEWIDX
#undef NEWFULL

#define SENDDATA r1
#define N_ITEMS r2
#define ATOMIC_PTR r3

#define HOWEMPTY r4
#define MUTEX r5
#define MUTEX_HOWEMPTY r5:4

#define NEWMUTEX r7
#define NEWEMPTY r6
#define NEWMUTEX_EMPTY r7:6

#define SENDIDX r10
#define NEWIDX r11

	.text
	.global nn_pipe_send_multi_fastpath
nn_pipe_send_multi_fastpath:
	{
		ELEMENTS_DATAPTR = memd(PIPE+#OFFSET_ELEMENTS_DATAPTR)
		SPINS = #31
		NEWMUTEX = #1
		ATOMIC_PTR = add(PIPE,#OFFSET_MUTEX_HOWEMPTY)
	}
	{
		loop0(.Lsend_copy_loop,N_ITEMS)
		p1 = cmp.eq(N_ITEMS,#0)
		if (p1.new) jump:nt .Lsend_done
	}
.Lsend_busy_spin:
	{
		MUTEX_HOWEMPTY = memd_locked(ATOMIC_PTR)
	}
	{
		p0 = cmp.eq(MUTEX,#0)
		p0 = !cmp.gt(N_ITEMS,HOWEMPTY)
		if (!p0.new) jump:nt .Lsend_busy
		NEWEMPTY = sub(HOWEMPTY,N_ITEMS)
	}
	{
		memd_locked(ATOMIC_PTR,p0) = NEWMUTEX_EMPTY
	}
	// OK, here we have the lock and there's enough room in the pipe
	// Copy in n_items of data
	{
		if (!p0) jump:nt .Lsend_busy_spin
		SENDIDX = memw(PIPE+#OFFSET_SENDIDX)
	}
	.falign
.Lsend_copy_loop:
	{
		DATA = memd(SENDDATA++#8)
		dcfetch(SENDDATA+#64)
		NEWIDX = add(SENDIDX,#1)
	}
	{
		memd(DATAPTR+SENDIDX<<#3) = DATA
		p0 = cmp.eq(NEWIDX,ELEMENTS)
		if (p0.new) SENDIDX = #0
		if (!p0.new) SENDIDX = NEWIDX
	}:endloop0
	// Write send_idx
	memw(PIPE+#OFFSET_SENDIDX) = SENDIDX
	// Unlock mutex
.Lsend_unlock_spin:
	MUTEX_HOWEMPTY = memd_locked(ATOMIC_PTR)
	{
		p0 = cmp.eq(MUTEX,#1)
		if (!p0.new) jump:nt .Lsend_someone_blocked_mutex
		MUTEX = #0
	}
	memd_locked(ATOMIC_PTR,p0) = MUTEX_HOWEMPTY
	if (!p0) jump:nt .Lsend_unlock_spin
	// increment howfull
	{
#if OFFSET_HOWFULL != 0
#error fixme: offset assumed
#endif
		r1:0 = combine(N_ITEMS,PIPE)
		jump nn_sem_add_fastpath
	}
.Lsend_done:
	jumpr r31
.Lsend_busy:
	pause(#10)
	{
		SPINS = add(SPINS,#-1)
		p0 = cmp.eq(SPINS,#0)
		if (!p0.new) jump:t .Lsend_busy_spin
	}
	{
		jump nn_pipe_send_multi_slowpath
	}
.Lsend_someone_blocked_mutex:
	{
		allocframe(#8)
		DATA = combine(N_ITEMS,PIPE)
	}
	memd(r29+#0) = DATA
	r0 = add(PIPE,#OFFSET_MUTEX)
	call nn_mutex_unlock_slowpath
	r1:0 = memd(r29+#0)
	deallocframe
	jump nn_sem_add_fastpath
	.size nn_pipe_send_multi_fast,.-nn_pipe_send_multi_fast
