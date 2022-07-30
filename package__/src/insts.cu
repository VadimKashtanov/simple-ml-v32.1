#include "package/head/insts.cuh"

//================ Insts =====================

uint inst_params[INSTS] = {
	8,	//dot1d
	9,	//dot2d
	10,	//kconvl33samepool22max
	3, 	//softmax
	7,	//lstm1d
	8,	//lstm2d
	4,	//gaussfiltre1d
	5,	//gaussfiltre2d
	9,	//dot1drecurent
	10,	//dot2drecurent
	8 	//dotgaussfiltre2d
};

const char* inst_name[INSTS] = {
	"dot1d",
	"dot2d",
	"kconvl33samepool22",
	"softmax",
	"lstm1d",
	"lstm2d",
	"gaussfiltre1d",
	"gaussfiltre2d",
	"dot1drecurent",
	"dot2drecurent",
	"dotgaussfiltre2d"
};

//============== Insts ==================

static const char* dot1d_params_names[8] = {
	"Ax", "Yx", 
	"activ", 
	"input_start", "ystart", "wstart", "locdstart",
	"drop_rate"
};

static const char* dot2d_params_names[9] = {
	"Ax", "Ay", "Bx", 
	"activ", 
	"input_start", "ystart", "wstart", "locdstart",
	"drop_rate"
};

static const char* kconvl33samepool22max_params_names[10] = {
	"Ax", "Ay",
	"n0", "n1",
	"activ",
	"input_start", "ystart", "wstart", "locdstart",
	"drop_rate"
};

static const char* softmax_params_names[3] = {
	"len", "input_start", "ystart"
};

static const char* lstm1d_params_names[0] = {
	"X","Y", "istart","ystart","wstart","locdstart", "drop_rate"
};

static const char* lstm2d_params_names[0] = {
	"Ax","Ay","Bx", "istart","ystart","wstart","locdstart", "drate"
};

static const char* gaussfiltre1d_params_names[0] = {
	"len", "istart","ystart","wstart"
};

static const char* gaussfiltre2d_params_names[0] = {
	"X","Y", "istart","ystart","wstart"
};

static const char* dot1drecurent_params_names[0] = {
	"Ax","At", "Yx", "activ", "istart","ystart","wstart","lstart", "drate"
};

static const char* dot2drecurent_params_names[0] = {
	"Ax","Ay","At","Bx","activ","istart","ystart","wstart","lstart", "drate"
};

static const char* dotgaussfiltre2d_params_names[0] = {
	"Ax","Ay", "Bx", "istart","ystart","wstart","locdstart", "drate"
};

const char** inst_param_name[INSTS] = {
	dot1d_params_names,
	dot2d_params_names,
	kconvl33samepool22max_params_names,
	softmax_params_names,
	lstm1d_params_names,
	lstm2d_params_names,
	gaussfiltre1d_params_names,
	gaussfiltre2d_params_names,
	dot1drecurent_params_names,
	dot2drecurent_params_names,
	dotgaussfiltre2d_params_names
};

check_f inst_check[INSTS] = {
	dot1d_check,
	dot2d_check,
	kconvl33samepool22max_check,
	softmax_check,
	lstm1d_check,
	lstm2d_check,
	gaussfiltre1d_check,
	gaussfiltre2d_check,
	dot1drecurent_check,
	dot2drecurent_check,
	dotgaussfiltre2d_check
};

cpu_f INST_CPU[INSTS] = {
	dot1d_cpu,
	dot2d_cpu,
	kconvl33samepool22max_cpu,
	softmax_cpu,
	lstm1d_cpu,
	lstm2d_cpu,
	gaussfiltre1d_cpu,
	gaussfiltre2d_cpu,
	dot1drecurent1d_cpu,
	dot2drecurent2d_cpu,
	dotgaussfiltre2d_cpu
};

use_f INST_USE[INSTS] = {
	dot1d_use,
	dot2d_use,
	kconvl33samepool22max_use,
	softmax_use,
	lstm1d_use,
	lstm2d_use,
	gaussfiltre1d_use,
	gaussfiltre2d_use,
	dot1drecurent_use,
	dot2drecurent_use,
	dotgaussfiltre2d_use
};

train_f INST_FORWARD[INSTS] = {
	dot1d_forward,
	dot2d_forward,
	kconvl33samepool22max_forward,
	softmax_forward,
	lstm1d_forward,
	lstm2d_forward,
	gaussfiltre1d_forward,
	gaussfiltre2d_forward,
	dot1drecurent_forward,
	dot2drecurent_forward,
	dotgaussfiltre2d_forward
};

train_f INST_BACKWARD[INSTS] = {
	dot1d_backward,
	dot2d_backward,
	kconvl33samepool22max_backward,
	softmax_backward,
	lstm1d_backward,
	lstm2d_backward,
	gaussfiltre1d_backward,
	gaussfiltre2d_backward,
	dot1drecurent_backward,
	dot2drecurent_backward,
	dotgaussfiltre2d_backward
};