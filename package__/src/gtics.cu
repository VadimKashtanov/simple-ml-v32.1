#include "package/head/gtics.cuh"

//		Make
Gtic_t * gtic_mk(Opti_t * opti, uint gtic_algo) {
	Gtic_t * ret = (Gtic_t*)malloc(sizeof(Gtic_t));

	//	Opti based on
	ret->opti = opti;

	//	Genetic tmpt used for selection
	ret->gtic_algo = gtic_algo;
	ret->gtic_space = GTIC_MK_ARRAY[gtic_algo](ret);

	return ret;
};

//		Select sets
void gtic_select(Gtic_t * gtic) {
	GTIC_SELECT_ARRAY[gtic->gtic_algo](gtic);
};

void gtic_set_args(Gtic_t * gtic, uint argc, char ** argv) {
	GTIC_SET_ARGS_ARRAY[gtic->gtic_algo](gtic, argc, argv);
};

//		Free
void gtic_free(Gtic_t * gtic) {
	GTIC_FREE_ARRAY[gtic->gtic_algo](gtic);
	free(gtic);
};

//================================================================

//	Build Gtic space
void* (*GTIC_MK_ARRAY[GTICS])(Gtic_t * gtic) = {
	gtic_mk_elite,
	gtic_mk_genetique1
};

//	Compute Score and rank it
void (*GTIC_SELECT_ARRAY[GTICS])(Gtic_t * gtic) = {
	gtic_select_elite,
	gtic_select_genetique1
};

//	Set Argv
void (*GTIC_SET_ARGS_ARRAY[GTICS])(Gtic_t * gtic, uint argc, char ** argv) = {
	gtic_set_args_elite,
	gtic_set_args_genetique1
};

//	Free the structure
void (*GTIC_FREE_ARRAY[GTICS])(Gtic_t * gtic) = {
	gtic_free_elite,
	gtic_free_genetique1
};