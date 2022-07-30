#include "package/head/optis.cuh"

/*		-------------    Build  ------------ */
Opti_t * opti_mk(Train_t * train, uint score_algo, uint opti_algo) {
	if (score_algo >= SCORES)
		ERR("Score number %li doesn't exists. Max is %li", score_algo, SCORES - 1)
	if (opti_algo >= OPTIS)
		ERR("Opti number %li doesn't exists. Max is %li", opti_algo, OPTIS - 1)

	Opti_t * ret = (Opti_t*)malloc(sizeof(Opti_t));
	
	ret->train = train;

	//	Cpu ram arrays
	ret->set_score = (float*)malloc(sizeof(float) * train->sets);
	ret->set_rank = (uint*)malloc(sizeof(uint) * train->sets);

	//	Gpu vram arrays
	SAFE_CUDA(cudaMalloc((void**)&ret->set_score_d, sizeof(float) * train->sets));
	SAFE_CUDA(cudaMalloc((void**)&ret->set_rank_d, sizeof(uint) * train->sets));

	//	Algorithms
	ret->score_algo = score_algo;
	ret->opti_algo = opti_algo;

	ret->score_space = OPTI_SCORE_SPACE_MK_ARRAY[score_algo](ret);
	ret->opti_space = OPTI_OPTI_SPACE_MK_ARRAY[opti_algo](ret);

	return ret;
};

/*		-------------    Compute Score  ------------ */
bool is_sorted(float * scores, uint * podium, uint n) {
	for (uint i=1; i < n; i++)
		if (scores[podium[i-1]] > scores[podium[i]])
			return false;
	return true;
};

void opti_score(Opti_t * opti) {
	//	============== Compute Scores ============
	OPTI_COMPUTE_SCORE_ARRAY[opti->score_algo](opti);

	//	============== Compute rank ==============
	
	uint sets = opti->train->sets;

	float * scores = opti->set_scores;
	float podium[sets];	//on 0th place is the best set (ca peut tres bien etre 32, 4 ou 0)

	for (uint i=0; i < sets; i++)
		podium[i] = i;

	//Rank score
	uint c;
	while (! is_sorted(scores, podium, sets) ) {
		for (uint i=1; i < sets; i++) {
			if (scores[podium[i-1]] > scores[podium[i]]) {
				c = podium[i];
				podium[i] = podium[i-1];
				podium[i-1] = c;
			}
		}
	}

	//	On podium are sorted sets. podium[0] == id of the best set
	//	Put in set_rank. So set_rank[i] = rank of i'th set
	for (uint i=0; i < sets; i++) {
		//	The i'th place on podium is set to set_rank[podium[i]]
		opti->set_rank[podium[i]] = i; 
	}

	SAFE_CUDA(cudaMemcpy(opti->set_rank_d, opti->set_rank, sizeof(uint) * sets, cudaMemcpyHostToDevice));

	//	Build podium
	for (uint i=0; i < sets; i++)
		opti->podium[opti->set_rank[i]] = i;	//in set_rank sets are stored in order from 0th set to last, and each case have the position on podium
};

/*		-------------    Optimize  ------------ */
void opti_dloss(Opti_t * opti) {
	OPTI_SCORES_DLOSS_ARRAY[opti->score_algo](opti);
};

void opti_opti(Opti_t * opti) {
	OPTI_OPTIMIZE_ARRAY[opti->opti_algo](opti);
};

/*		-------------    Free structure  ------------ */
void opti_free(Opti_t * opti) {
	free(opti->scores);
	free(opti->rank);

	SAFE_CUDA(cudaFree(opti->scores_d));
	SAFE_CUDA(cudaFree(opti->rank_d));

	OPTI_FREE_SCORE_ARRAY[opti->score_algo](opti);
	OPTI_FREE_OPTI_ARRAY[opti->opti_algo](opti);

	free(opti);
};

//	-----------------------------------------------------------------------

//	Mk score
void* (*OPTI_SCORE_SPACE_MK_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_space_mk,
	CROSSENTROPY_space_mk
};

//	Mk opti
void* (*OPTI_OPTI_SPACE_MK_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_space_mk,
	MOMENTUM_space_mk,
	RMSPROP_space_mk,
	ADAM_space_mk,
	ADADELTA_space_mk,
	ADAMAX_space_mk
};

//	Score
void* (*OPTI_COMPUTE_SCORE_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_score,
	CROSSENTROPY_score
};

//	Score DLOSS
void (*OPTI_SCORES_DLOSS_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_dloss,
	CROSSENTROPY_dloss
};

//	Optimize
void* (*OPTI_OPTIMIZE_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_optimize,
	MOMENTUM_optimize,
	RMSPROP_optimize,
	ADAM_optimize,
	ADADELTA_optimize,
	ADAMAX_optimize
};

//	Free score
void* (*OPTI_FREE_SCORE_ARRAY[SCORES])(Opti_t * opti) = {
	MEANSQUARED_free,
	CROSSENTROPY_free
};

//	Free opti
void* (*OPTI_FREE_OPTI_ARRAY[OPTIS])(Opti_t * opti) = {
	SGD_optimize,
	MOMENTUM_free,
	RMSPROP_free,
	ADAM_free,
	ADADELTA_free,
	ADAMAX_free
};