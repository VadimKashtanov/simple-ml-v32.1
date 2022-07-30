#include "package/package.cuh"

/*
                   1       2       3       4     5    6       7
./optimize_mdl data.bin mdl.bin out.bin echopes sets score optimizer
	
	Score : me, ce
	Optimizer : sgd, moment, rmsprop, adam, adadelta, adamax

*/

#define PARAMS 4
//								0      1      2        3         4       5       6
char * parametres[PARAMS] = {"data", "mdl", "out", "echopes", "sets", "score", "opti"};
char * arguments[PARAMS] =  {  0,      0,     0,       0,        0,       0,      0  };

char *_scores_arr[SCORES] = {"me", "ce"};
char *_optis_arr[OPTIS] = {"sgd", "moment", "rmsprop", "adam", "adadelta", "adamax"};

uint get_by_name(char * name, char ** arr, uint len) {
	for (uint i=0; i < len; i++) {
		if (strcmp(name, arr[i]) == 0)
			return i;
	}
	ERR("%s is not recognized", name);
	return;
};

int main(int argc, char ** argv) {
	if (argc == 6) {

		//// Parse Arguments
		etc_parse_arguments(argc, argv, paramc, parametres, arguments);

		//// Sources
		char * datafile = arguments[0];
		FILE * mdlfp = open(arguments[1], "rb");
		char * outfile = arguments[2];

		//// Trainning setup
		uint echopes = atoi(arguments[3]);
		uint sets = atoi(arguments[4]);

		//// Optimisation configuration
		uint score = get_by_name(arguments[5], _scores_arr, SCORES);
		uint opti = get_by_name(arguments[6], _optis_arr, OPTIS);

		//// Load to Ram and Vram
		Data_t * data = data_load(datafile);
		Mdl_t * mdl = mdl_fp_load(mdlfp);
		fclose(mdlfp);

		//// Build Train_t and Opti_t
		Train_t * train = mk_train(mdl, data, sets);
		train_random_weights_form_mdl(train);
		Opti_t * opti = opti_mk(train, score, opti);

		uint start_seed;

		for (uint lp=0; lp < echopes; lp++) {
			//	Load a batch
			data_load_batch(data, datafile, rand() % data->batchs);

			//	Initialise correctly
			train_set_input(train);
			train_null_grad_meand(train);

			//	Forward - Backward
			start_seed = rand() % 100000;

			train_forward(train, start_seed);
			opti_dloss(opti);
			train_backward(train, start_seed);

			//	Optimize
			opti_opti(opti);
		};

		////	Compute Score
		opti_score(opti);

		printf("## Scores ##\n");
		for (uint i=0; i < sets; i++) {
			printf("%i| %.5g (set id : %i)\n", i, opti->set_score[podium[i]], podium[i]);
		}

		//	Take Best set
		uint best_set = opti->podium[0];

		train_cpy_ws_to_mdl(train, best_set);

		mdlfp = open(arguments[2], "wb");
		mdl_fp_write(mdl, mdlfp);
		fclose(mdlfp);

		//	Free all to make a correct valgrind and juste to make all clean (and each malloc have to be freed)
		opti_free(opti);
		train_free(train);
		data_free(data);
		mdl_free(mdl);
	} else {
		ERR("Not 5 arguments")
	}
};