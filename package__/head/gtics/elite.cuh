#pragma once

#include "kernel/head/gtics.cuh"

/*
============|===================================================================|
	Elite	|	Juste take a % of best sets, and fill the others with bests.	|
			|	If there is 32 sets and 8 eltes									|
			|		so the upgradable is 32-8=24 								|
			|		and 24/8 == 3, so each elite generate 3 clone 				|
			|	You have to chose elites as (sets-elites)/elites is a natural	|
============|===================================================================|
*/

uint gtic_elite_elites = 1;
uint gtic_elite_echopes = 1;

typedef struct {
	uint elites;	//
	uint portion;	//how many clones each elite make

	//	Could store an `float` array of sets*weights size, to store the new generation, and then copy to train->_weight
	//	But I can juste alloc it when I need it. When I will have a bigger Vram, I will alloc it here to not alloc/free always
} GticElite_t;

void * gtic_mk_elite(Gtic_t * gtic);
void gtic_select_elite(Gtic_t * gtic);
void gtic_free_elite(Gtic_t * gtic);

void gtic_set_args_elite(Gtic_t * gtic, uint argc, char ** argv);	//have to be in order and each have to be set