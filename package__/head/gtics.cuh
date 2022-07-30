#pragma once

/*	Genetics - Select sets.
*/

/*	Gtic 			|   Description																	|
====================|===============================================================================|
	Elite			|	Juste take a % of best sets, and fill the others with bests.				|
					|	If there is 32 sets and 8 eltes												|
					|		so the upgradable is 32-8=24 											|
					|		and 24/8 == 3, so each elite generate 3 clone 							|
					|	You have to chose elites as (sets-elites)/elites is a natural				|
====================|===============================================================================|
	Genetique 1 	|   Premiere Tentative de construction d'un systeme simple de genetique.		|
					|	Base sur la genetique de la vie et donc sur ca sagesse d'ou qu'elle vienne.	|	
====================|===============================================================================|
*/

#include "package/head/gtics/elite.cuh"
#include "package/head/gtics/genetique_1.cuh"

//	Build Gtic space
extern void* (*GTIC_MK_ARRAY[GTICS])(Gtic_t * gtic);

//	Compute Score and rank it
extern void (*GTIC_SELECT_ARRAY[GTICS])(Gtic_t * gtic);
extern void (*GTIC_SET_ARGS_ARRAY[GTICS])(Gtic_t * gtic, uint argc, char ** argv);

//	Free the structure
extern void (*GTIC_FREE_ARRAY[GTICS])(Gtic_t * opti);