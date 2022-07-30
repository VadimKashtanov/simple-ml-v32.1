#pragma once

/*
	First genetic algo life-like

*/

void * gtic_mk_genetique1(Gtic_t * gtic);
void gtic_select_genetique1(Gtic_t * gtic);
void gtic_free_genetique1(Gtic_t * gtic);

void gtic_set_args_genetique1(Gtic_t * gtic, uint argc, char ** argv);	//have to be in order and each have to be set