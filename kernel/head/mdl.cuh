#pragma once

#include "kernel/head/etc.cuh"
#include "package/meta.cuh"

typedef struct {
	//Instructions stack
	uint insts;
	uint * id;
	uint ** param;

	//Data link context
	uint inputs;
	uint outputs;
	
	//Context for computation
	uint vars, weights, locds;
	uint total;	//inputs + vars = inputs + none_output_vars + outputs

	//Weight
	float * weight;	//	cpu malloc()
	float * step;	//	gradient step of each weight (w += step*dw)
} Mdl_t;

//	I/O
Mdl_t* mdl_fp_load(FILE * fp);
void mdl_fp_write(Mdl_t * mdl, FILE * fp);

//	Ctrl
void mdl_check_correctness(Mdl_t * mdl);

//	Mem
void mdl_free(Mdl_t * mdl);	//useless

//	Plum
void mdl_print_inst(Mdl_t * mdl, uint inst);

extern uint inst_params[INSTS];
extern const char* inst_name[INSTS];
extern const char** inst_param_name[INSTS];

typedef void (*check_f)(uint* params);

extern check_f INST_CHECK[INSTS];