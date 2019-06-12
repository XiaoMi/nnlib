/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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
#ifndef NN_CONST_PREP_SHARE_H
#define NN_CONST_PREP_SHARE_H 1

#include <stdlib.h>
#include <stdint.h>
#include "nn_graph.h"
struct nn_cpshare_typedesc;
//
// This is a mechanism by which data 'prepared from const' - in most cases a reordered weight array -
// can be shared amongst multiple nodes, provided
//    - all such nodes have the same const node as input_iterator
//    - the nodes are of the same "class" (as defined by this mechanism). Generally that means that
//      the nodes are of the same type, or possibly are two variant types handled by the same code.
//    - other constraints may be imposed by the nodes using this mechanism, i.e. there may be cases
//      where another instance of the same class may want to prepare different data from the same
//      const.
//
// How it works:
//   - the node defines a 'class' which is a subclass of struct nn_cpshare_base; meaning
//      you define a struct which starts with the same fields (and could be the same as) 
//      nn_cpshare_base.
//   - you declare a typedesc object: a const variable of type nn_cpshare_typedesc; this currently just contains
//     the size of the type, and a pointer to a dtor (which may be NULL for default dtor).
//
//   then you have code like this, in the 'check', or first exec method:
//
//       struct node  *const_node = nn_cpshare_get_const_node( nn, self, input_no );
//		 // that is the pointer to the Const which is attached to the input. If it's NULL, it means
//       // the node couldn't be found, or isn't const; both are likely fatal.
//       struct my_cpshare_type * cpshare= 
//              (struct my_cpshare_type*) nn_cpshare_get_existing( nn, &my_cp_typedesc, const_node );
//       // if the Const already has a object of *your* type attached to it, cpshare will 
//       // be non-null and point to it. Its reference count has been incremented.
//       if( cpshare == NULL ){
//             // either there is no attached object, or it's got the wrong type; make a new one.
//             cpshare = (struct my_cpshare_type*)nn_cpshare_new( nn, &my_cp_typedesc);
//             //that makes a new object, cleared to 0, with the header filled in. NULL return is fatal.
//             //
//             ... proceed to do your const preparation, attaching newly allocated data to it.
//             //
//             nn_cpshare_attach( nn, const_node, cpshare );
//             // the 'attach' call will attempt to attach the new object to the Const; if there's not
//             // one already, it will be attached and the reference count bumped.
//        }
//        // now, hang on to the 'cpshare' pointer and use it from run to run.
//        // Eventually, when the node is destroyed:
//        nn_cpshare_decfref( nn, cpshare );
//        
//        // the dtor of Const will also do a 'decref' on its pointer, if not null. The decref
//        which takes the reference count to zero will call your type's destructor.
//
// Note: In some cases, you may get a non-null pointer from nn_cpshare_get_existing, and not be
// able to use because e.g. the preparation may depend on something outside the const.
// In that case, you should have fields in your type which you can look at to see if it's
// usable in other nodes, and add the following before if(cpshare ==NULL):
//
//        while( cpshare != NULL && ! i_can_use(cp_share)){
//            cpshare = (struct my_cpshare_type*)
//                      nn_cpshare_get_another_existing( nn, &my_cp_typedesc, cpshare);
//        }
// This function removes a reference from *cpshare, and either returns NULL, or a different
// attached instance (increfing it). Currently we only attach one, so it will always return NULL.
// But if you follow the logic above, the code will work if & when we support more than one.
//
//     nn_cpshare_base has the following fields:
//
#define NN_CPSHARE_HEADER \
	struct nn_cpshare_typedesc const * typedesc;\
    int32_t ref_count;\
	void *ptr_w;			/* e.g. for prepared weights*/\
	void *ptr_sumb;			/* e.g. for prepared 'sumb'*/\
	void *ptr_x;			/* generic pointer */

struct nn_cpshare_base {
	NN_CPSHARE_HEADER
};
// The  nn_cpshare_new() will set up 'typedesc', set ref_count to 1, and clear rest of the struct.
// The default destructor (selected by a null pointer in the typedesc) will simply nn_free() any of the
// three pointers which are NULL. You can make your  own dtor by defining it in the type desc object
typedef void(*nn_cpshare_dtor_fp )( struct nn_graph*, struct nn_cpshare_base*);

struct nn_cpshare_typedesc {
	unsigned size;
	nn_cpshare_dtor_fp dtor;
};

// Here's how to define your own type:
//
//  struct my_cpshare_type {
//     NN_CPSHARE_HEADER		// must have this
//     int some_other_field;
//     uint8_t * some_ptr;
//  }
//  static const struct nn_cpshare_typedesc 
//    my_cp_typedesc = { sizeof(struct my_cpshare_type) };
//
//  (or, if you  want a dtor ):
//
//  static const struct nn_cpshare_typedesc my_cp_typedesc = {
//			.size = sizeof(struct my_cpshare_type),
//			.dtor = my_dtor
//
// If you define a dtor, you are also responsible for freeing ptr_w, ptr_sumb, ptr_x if they
// are used.
//

// Be careful with the 'void*' parameters in the below - all of these must point to
// things that can be treated as struct nn_cpshare_base.
//
// (functions which are commented out, are inlined below)
struct nn_node* nn_cpshare_get_const_node( struct nn_graph *nn, struct nn_node* self, int input_no );
//struct nn_cpshare_base *nn_cpshare_new( struct nn_graph * nn, struct nn_cpshare_typedesc const* );
void nn_cpshare_decref( struct nn_graph *nn, void * cpshare );
struct nn_cpshare_base *nn_cpshare_get_existing( struct nn_graph * nn,
	struct nn_cpshare_typedesc const*, struct nn_node const * );
//struct nn_cpshare_base *nn_cpshare_get_another_existing( struct nn_graph * nn,
//	struct nn_cpshare_typedesc const*, void * cpshare );
void nn_cpshare_attach( struct nn_graph *nn, struct nn_node* const_node, void * cpshare );


static inline struct nn_cpshare_base *
nn_cpshare_new( struct nn_graph * nn, struct nn_cpshare_typedesc const* td)
{
	struct nn_cpshare_base *p= (struct nn_cpshare_base *)nn_calloc( 1,td->size);
	if(p != NULL){
		p->typedesc = td;
		p->ref_count =1;
	}
	return p;
}

static inline struct nn_cpshare_base *
nn_cpshare_get_another_existing( struct nn_graph * nn,
	struct nn_cpshare_typedesc const* tdp, void * cpsharev )
{
	// currently this will always fail - but
	// but we can add that later in the same API.
	nn_cpshare_decref( nn, cpsharev );
	return NULL;
}
#endif // NN_CONST_PREP_SHARE_H
