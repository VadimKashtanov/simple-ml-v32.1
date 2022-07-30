#include "pkg_head/insts/softmax/softmax_th32.cuh"

void softmax_use_call_mode_th32(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint len=mdl->param[inst][0],			\
		 input_start=mdl->param[inst][1],	\
		 ystart=mdl->param[inst][2];

	if (len <= 32) {
		softmax_use_th32<<<dim3(1),dim3(32)>>>(
			len,
			time,
			mdl->total,
			input_start, ystart,
			use->var);
		//cudaDeviceSynchronize();
	} else {
		ERR("Can't handl more than 32 pixels for softmax")
	}
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void softmax_forward_call_mode_th32(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint len=mdl->param[inst][0],			\
		 input_start=mdl->param[inst][1],	\
		 ystart=mdl->param[inst][2],
		 locdstart=mdl->param[inst][3];

	if (len <= 32) {
		softmax_forward_th32<<<dim3(train->sets),dim3(32)>>>(
			len,
			time,
			mdl->total, mdl->locds,
			input_start, ystart, locdstart,
			train->sets,
			train->_var);
		//cudaDeviceSynchronize();
	}
};

//-------------------------- backward ---------------------

void softmax_backward_call_mode_th32(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint len=mdl->param[inst][0],			\
		 input_start=mdl->param[inst][1],	\
		 ystart=mdl->param[inst][2],
		 locdstart=mdl->param[inst][3];

	if (len <= 32) {
		softmax_backward_th32<<<dim3(train->sets),dim3(32)>>>(
			len, 
			time,
			mdl->total, mdl->locds,
			input_start, ystart, locdstart,
			train->sets,
			train->_var, train->_grad);
		//cudaDeviceSynchronize();
	}
};