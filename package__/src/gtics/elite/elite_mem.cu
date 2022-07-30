#include "package/head/gtics/elite/elite.cuh"

void * gtic_mk_elite(Gtic_t * gtic) {
	GticElite_t * ret = (GticElite_t*)malloc(sizeof(GticElite_t));

	ret->elites = gtic_elite_elites;
	ret->portion = (gtic->opti->train->sets - ret->elites)/ret->elites;

	return (void*)ret;
};

void gtic_free_elite(Gtic_t * gtic) {
	free(gtic->gtic_space);
};

void gtic_set_args_elite(Gtic_t * gtic, uint argc, char ** argv) {
	if (argc == 2) {
		gtic_elite_elites = atoi(argv[0]);
		gtic_elite_echopes = atoi(argv[1]);

	} else {
		ERR("Elite : Il faut 2 arguments pas (%li)", argc);
	}
};