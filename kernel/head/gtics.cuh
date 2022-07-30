#pragma once

#include "kernel/head/optis.cuh"

//	Constants and parameters for genetic algorithms are declared in package/package.cuh

typedef struct {
	Opti_t * opti;

	//puis on select les sets sur le ranking et le score.
	//	Peut etre aussi des ranking systemes differents.
	//	Basees sur plusieurs choses

	//	Gtics
	uint gtic_algo;
	void * gtic_space;
} Gtic_t;

//		Mem
//	Defined in package/
Gtic_t * gtic_mk(Opti_t * opti, uint gtic_algo);

//		Controle
//	Defined in package/
void gtic_select(Gtic_t * gtic);
void gtic_set_args(Gtic_t * gtic, uint argc, char ** argv);

//		Free
//	Defined in package/
void gtic_free(Gtic_t * gtic);