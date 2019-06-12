
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

/*
 * Portable Operating System
 * We assume we have nn_futex_wait and nn_futex_wake ... and that's it!
 */
#include <nn_graph.h>
uint64_t nn_sem_add_slowpath_withret(nn_sem_t *sem, int amt, uint64_t ret)
{
	/* Atomic add amt to sem */
	nn_atomic_add32(&sem->raw,amt);			// We're assuming amt is reasonable and not negative...
	/* nn_futex_wake(sem,amt); */
	if (sem->waiters) nn_futex_wake(&sem->as_futex,amt);	// possibly unnecessary, this *is* the slowpath.
	return ret;
}

void nn_sem_add_slowpath(nn_sem_t *sem, int amt)
{
	nn_sem_add_slowpath_withret(sem,amt,0);
}

void nn_sem_sub_slowpath(nn_sem_t *sem, int amt)
{
	uint32_t expected;
	uint32_t new;
	uint32_t amt_avail;
	while (amt > 0) {
		expected = sem->raw;
		amt_avail = expected & 0x0FFFF;
		if (amt_avail == 0) {
			new = expected + 0x10000;
			if (nn_atomic_casu32(&sem->raw,expected,new) != expected) {
				// We couldn't increment waiter, so just restart
				continue;
			}
			expected = new;
			while (1) {
				/* While the value in the semaphore is zero, go to sleep */
				if (amt_avail == 0) nn_futex_wait(&sem->as_futex,expected);
				if ((amt_avail = ((expected = sem->raw) & 0x0FFFF)) == 0) continue;	// nothing there, go back to sleep
				new = expected - 0x10001;				// try to decrement waiters and count
				if (nn_atomic_casu32(&sem->raw,expected,new) != expected) {
					/* Something changed, but we're still showing waited, retry */
					expected = sem->raw;
					continue;
				} else {
					/* We decremented waiters and count */
					amt -= 1;
					break;
				}
			}
		} else {
			/* Don't mark sleeping yet, just try to decrement count */
			uint32_t amt_to_decrement = Q6_R_minu_RR(amt,amt_avail);
			new = expected - amt_to_decrement;
			if (nn_atomic_casu32(&sem->raw,expected,new) == expected) {
				amt -= amt_to_decrement;
			}
		}
	}
}

void nn_mutex_lock_slowpath(nn_mutex_t *mutex)
{
	uint32_t old;
	while (1) {
		/* change lock to "maybe waiters".  If was 0, we're owner */
		old = mutex->raw;
		if ((nn_atomic_casu32(&mutex->raw,old,2)) == old) {
			/* Successfully updated mutex to 2 (waiters) */
			if (old == 0) return;
			/* Else successfully set to two, wait for change */
			nn_futex_wait(&mutex->as_futex,2);
		} 
		/* Else, atomic update failed, try again */
	}
}

void nn_mutex_unlock_slowpath(nn_mutex_t *mutex)
{
	uint32_t old = mutex->raw;
	/* Swap 0 into lock */
	if (nn_atomic_casu32(&mutex->raw,old,0) != old) {
		/* Oops! atomic update failed, try again */
		return nn_mutex_unlock_slowpath(mutex);
	}
	/* If old value 1 (or somehow unlocked already...), done */
	if (old < 2) return;
	/* Else, nn_futex_wake(mutex,1); */
	nn_futex_wake(&mutex->as_futex,1);
}

