#pragma once

#include "kernel/head/data.cuh"
#include "kernel/head/mdl.cuh"

//	Locd is very usefull for max finding. Beacause, otherwise you will have to re-compute all the forward

typedef struct train_model {
	//	Model and sets trainned
	Mdl_t * mdl;
	uint sets;
	
	//	Data
	Data_t * data;

	//	Vram arrays
	float * _weight;	//_weight[sets ][wsize]
	float * _var;		//   _var[times][sets ][vsize]
	float * _locd;		//  _locd[times][sets ][lsize]
	float * _grad;		//  _grad[times][sets ][vsize]
	float * _meand;		// _meand[sets ][wsize]
} Train_t;

//	Mem
Train_t* mk_train(Mdl_t * mdl, Data_t * data, uint sets);
void train_random_weights(Train_t * train);
void train_random_weights_from_mdl(Train_t * train);
void train_cpy_ws_to_mdl(Train_t * train, uint set);
Train_t * extract_to_new_train(Train_t * old, uint amount, uint * set_id);	//extract set[0], set[4], set[2], set[1] and set[23]  to the new train and in this order

//pas obliatoire
//void train_snapshot(Train_t * train);	//////////////////////////!!!!!!!!!!!!!!!!!!!!!!!

//	Controle
void train_set_input(Train_t * train);
void train_null_grad_meand(Train_t * train);
void train_forward(Train_t * train, uint start_seed);
void train_backward(Train_t * train, uint start_seed);

//	Free
void train_free(Train_t * train);

typedef void (*train_f)(Train_t* train, uint inst, uint time, uint start_seed);
extern train_f INST_FORWARD[INSTS];
extern train_f INST_BACKWARD[INSTS];

/*	Test all
uint	models	 	x1
[models]
	Mdl_t	model 		x1
	uint	lines 		x1
	uint 	sets 		x1
	uint 	echopes 	x1
	Sep_t	vsep		x1
	Sep_t	wsep		x1
	Sep_t	lsep		x1

	float	weight		xweights*sets
	[optis]
		uint	opti_args	x1
		[opti_args]						#opti arguments are in STR, we will check the corectness of parsing of c/cuda
			uint	len 		x1
			char	char		xlen
		[end]

		[echopes]
			float	inp			xinps*lines
			float	out 		xouts*lines
			float   var 		xvars*lines*sets
			float 	locd		xlocds*lines*sets
			float	grad		xvars*lines*sets
			float	meand		xweights*sets

			float	w 			xweights*sets
		[end]
	[end]

	float	weight		xweights*sets
	[gtics]
		uint	gtic_args	x1
		[gtic_args]						#gtic arguments are in STR, we will check the corectness of parsing of c/cuda		(from a terminal or a .json or python_test_file)
			uint	len 		x1
			char	char		xlen
		[end]

		[echopes]
			float 	scores		xsets
			uint 	rank		xsets

			float	inp			xinps*lines
			float	out 		xouts*lines
			float   var 		xvars*lines*sets
			float 	locd		xlocds*lines*sets

			float	w 			xweights*sets
		[end]
	[end]

	float	weight		xweights*sets
	[scores]
		[echopes]
			float 	scores		xsets
			uint 	rank		xsets

			float	inp			xinps*lines
			float	out 		xouts*lines
			float   var 		xvars*lines*sets
			float 	locd		xlocds*lines*sets
			float 	grad		xvars*lines*sets
			
			float	w 			xweights*sets
		[end]
	[end]
[end]
*/