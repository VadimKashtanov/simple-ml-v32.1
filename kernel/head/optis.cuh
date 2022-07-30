#pragma once

#include "kernel/head/train.cuh"

typedef struct optimizer_and_score {
	//	Train_t of model
	Train_t * train;

	//	Ranking
	float * set_score;		//	score of i'th set
	float * set_score_d;
	uint * set_rank;		//set_rank[i] give the place on podium of i'th set
	uint * set_rank_d;
	uint * podium;

	//	Algorithms
	uint score_algo, opti_algo;
	void * score_space, * opti_space;
} Opti_t;

//		Mem
//	Defined in package/
Opti_t * opti_mk(Train_t * train, uint score_algo, uint opti_algo);

//		Controle
//	Defined in package/
void opti_score(Opti_t * opti);
void opti_dloss(Opti_t * opti);
void opti_opti(Opti_t * opti);

//		Free
//	Defined in package/
void opti_free(Opti_t * opti);