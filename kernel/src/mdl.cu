#include "kernel/head/mdl.cuh"

Mdl_t* mdl_fp_load(FILE * fp) {
	Mdl_t * ret = (Mdl_t*)malloc(sizeof(Mdl_t));

	/*			Instructions		*/
	fread(&ret->insts, sizeof(uint), 1, fp);

	ret->id = (uint*)malloc(sizeof(uint) * ret->insts);
	ret->param = (uint**)malloc(sizeof(uint*) * ret->insts);

	for (uint i=0; i < ret->insts; i++) {
		//Instruction Id
		fread(&ret->id[i], sizeof(uint), 1, fp);

		//Parameters
		ret->param[i] = (uint*)malloc(sizeof(uint) * inst_params[ret->id[i]]);
		fread(ret->param[i], sizeof(uint), inst_params[ret->id[i]], fp);
	}

	fread(&ret->inputs, sizeof(uint), 1, fp);
	fread(&ret->outputs, sizeof(uint), 1, fp);

	fread(&ret->vars, sizeof(uint), 1, fp);
	fread(&ret->weights, sizeof(uint), 1, fp);
	fread(&ret->locds, sizeof(uint), 1, fp);

	ret->weight = (float*)malloc(sizeof(float) * ret->weights);
	ret->step   = (float*)malloc(sizeof(float) * ret->weights);
	
	fread(ret->weight, sizeof(float), ret->weights, fp);
	fread(ret->step, sizeof(float), ret->weights, fp);

	ret->total = ret->inputs + ret->vars;
	
	return ret;
};

void mdl_fp_write(Mdl_t * mdl, FILE * fp) {
	fwrite(&ret->insts, sizeof(uint), 1, fp);

	for (uint i=0; i < ret->insts; i++) {
		//Instruction Id
		fwrite(&ret->id[i], sizeof(uint), 1, fp);
		fwrite(ret->param[i], sizeof(uint), inst_params[ret->id[i]], fp);
	}

	fwrite(&ret->inputs, sizeof(uint), 1, fp);
	fwrite(&ret->outputs, sizeof(uint), 1, fp);

	fwrite(&ret->vars, sizeof(uint), 1, fp);
	fwrite(&ret->weights, sizeof(uint), 1, fp);
	fwrite(&ret->locds, sizeof(uint), 1, fp);

	fwrite(ret->weight, sizeof(float), ret->weights, fp);
	fwrite(ret->step, sizeof(float), ret->weights, fp);
};

void mdl_free(Mdl_t * mdl) {
	//	Insts
	free(mdl->id);
	for (uint i=0; i < mdl->insts; i++)
		free(mdl->param[i]);
	free(mdl->param);

	//Ws
	free(mdl->weight);
	free(mdl->step);

	//
	free(mdl);
};

void mdl_print_inst(Mdl_t * mdl, uint i) {
	uint inst_id = mdl->id[i];
	printf("\033[30;1;46m %s\033[0m:\n", inst_name[inst_id]);
	for (uint p=0; p < inst_params[inst_id]; p++)
		printf("\t\033[42;1;42m%s\033[0m: %i\n", inst_param_name[inst_id][p], mdl->param[i][p]);
};

void mdl_check_correctness(Mdl_t * mdl) {
	if (mdl->insts > 1000)
		raise(SIGINT);

	for (uint i=0; i < mdl->insts; i++) {
		if (mdl->id[i] >= INSTS)
			raise(SIGINT);

		INST_CHECK[mdl->id[i]](mdl->param[i]);
	}

	if (mdl->output_start != mdl->total - mdl->outputs)
		raise(SIGINT);

	if (mdl->total != mdl->inputs + mdl->vars)
		raise(SIGINT);
};