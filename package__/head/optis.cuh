#pragma once

/*	Scores    |          Loss(w,g)        |        dLoss(w,g)/dg       	|
========================================================================|
 MeanSquared  |        (1/2)*(w - g)^2    |  		   g - w		   	|
========================================================================|
 CrossEntropy | w*ln(g) + (1-w)*ln(1-g)   |    (w - g)/[g*(1 - g)]	   	|
========================================================================|
*/

//	Aller voire tous les papiers avec les algos sur : ttps://paperswithcode.com/method/adamax

/*	Optimizers 		|	Equation										|
====================|===================================================|
	SGD 			|	w -= alpha*grad(w)								|
====================|===================================================|
	Momentum		| 	v = moment*v - alpha*grad(w)					|
					|	w -= v 											|
====================|===================================================|
	RMSprop 		|	v = beta*v + (1-beta)*grad(w)^2					|
					|	w -= alpha*grad(w)/sqrt(v)						|
====================|===================================================|
	Adam 			|	m = beta0*m + (1 - beta0)*grad(w)				|
					|	v = beta1*m + (1 - beta1)*grad(w)^2 			|
					| 													|
					|	_m = m / ( 1 - beta0^t )						|
					|	_v = v / ( 1 - beta1^t )		t is echope 	|
					|													|
					|	w -= alpha * _m / sqrt(_v + eta)				|
====================|===================================================|
	Adadelta 		|	m = beta0*m + (1 - beta0)*grad(w)^2				|	
					|	delta_w = - sqrt(v + 1e-8) / sqrt(m + 1e-8)		|
					|	v = beta1*v + (1 - beta1)*delta_w^2 			|
					|													|
					|	w -= delta_w									|
====================|===================================================|
	Adamax 			|	m = beta0*m + (1 - beta0)*grad(w)				|
					|	u = max(beta1 * u, abs(grad(w)))				|
					|													|
					|	w -= alpha * m / (u * (1 - beta1^t))			|
====================|===================================================|

//	To add :	Nag, Nadam, Ftrl, One Cycle

*/

#include "package/head/optis/optis/sgd.cuh"
#include "package/head/optis/optis/momentum.cuh"
#include "package/head/optis/optis/rmsprop.cuh"
#include "package/head/optis/optis/adam.cuh"
#include "package/head/optis/optis/adadelta.cuh"
#include "package/head/optis/optis/adamax.cuh"

#include "package/head/optis/scores/crossentropy.cuh"
#include "package/head/optis/scores/meansquared.cuh"

//	Build Score & Optimizer space
extern void* (*OPTI_SCORE_SPACE_MK_ARRAY[SCORES])(Opti_t * opti);
extern void* (*OPTI_OPTI_SPACE_MK_ARRAY[OPTIS])(Opti_t * opti);

//	Dloss of Train_t
extern void (*OPTI_SCORES_DLOSS_ARRAY[SCORES])(Opti_t * opti);

//	Optimize weights of Train_t
extern void (*OPTI_OPTIMIZE_ARRAY[OPTIS])(Opti_t * opti);

//	Compute Score and rank it
extern void (*OPTI_COMPUTE_SCORE_ARRAY[SCORES])(Opti_t * opti);

//	Free the structure
extern void (*OPTI_FREE_SCORE_ARRAY[SCORES])(Opti_t * opti);
extern void (*OPTI_FREE_OPTI_ARRAY[OPTIS])(Opti_t * opti);