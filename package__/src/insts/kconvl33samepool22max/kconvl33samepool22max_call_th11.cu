#include "pkg_head/insts/kconvl33samepool22max/kconvl33samepool22max_th11.cuh"

//======================= Use_t Forward ===========================

void kconvl33samepool22max_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;
	uint Ax=mdl->param[inst][0];
	uint Ay=mdl->param[inst][1];
	uint n0=mdl->param[inst][2];
	uint n1=mdl->param[inst][3];
	uint activ=mdl->param[inst][4];
	uint input_start=mdl->param[inst][5];
	uint ystart=mdl->param[inst][6];
	uint wstart=mdl->param[inst][7];

	//Copy Kernels to Constant memory
	kconvl33samepool22max_use_const_MemCpyToSymbol(use->weight + wstart, n0*n1*9);

	//	Kconvl with 'boundared' input image
	kconvl33samepool22max_use_const_th1x1<<<dim3(KERN_DIV(Ax/2,16), KERN_DIV(Ay/2,16), n1), dim3(16,16,1)>>>(
		n0, n1, Ax, Ay, 
		activ,
		time,
		mdl->total, mdl->weights,
		istart, wstart, ystart,
		use->var, use->weight);
};

//========================		Train_t	  =========================

//----------------------------- forward ---------------------------

void kconvl33samepool22max_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Ay=mdl->param[inst][1],			\
		 n0=mdl->param[inst][2],			\
		 n1=mdl->param[inst][3],			\
		 activ=mdl->param[inst][4],		\
		 input_start=mdl->param[inst][5],	\
		 ystart=mdl->param[inst][6],		\
		 wstart=mdl->param[inst][7],		\
		 locdstart=mdl->param[inst][8],	\
		 drop_rate_int=mdl->param[inst][9];

	float drop_rate = drop_rate_int / 100;

	uint seed;

	for (uint set=0; set < train->sets; set++) {
		seed = (uint)pseudo_randomi(start_seed + set*inst);

		kconvl33samepool22max_train_const_MemCpyToSymbol(train->_weight + set*(mdl->weights)+wstart, n0*n1*9);
		
		kconvl33samepool22max_forward_const_th1x1<<<dim3(KERN_DIV(Ax/2,16), KERN_DIV(Ay/2,16), n1), dim3(16,16,1)>>>(
			n0, n1, Ax, Ay,
			activ,
			time,
			mdl->total, mdl->weights, mdl->locds,
			istart, wstart, ystart, locdstart,
			seed, drop_rate,
			set, train->sets,
			train->_var, train->_weight, train->_locd);

		cudaDeviceSynchronize();
	}
};

//----------------------------- backward ---------------------------

void kconvl33samepool22max_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Ay=mdl->param[inst][1],			\
		 n0=mdl->param[inst][2],			\
		 n1=mdl->param[inst][3],			\
		 activ=mdl->param[inst][4],		\
		 istart=mdl->param[inst][5],	\
		 ystart=mdl->param[inst][6],		\
		 wstart=mdl->param[inst][7],		\
		 locdstart=mdl->param[inst][8],	\
		 drop_rate_int=mdl->param[inst][9];

	float drop_rate = drop_rate_int / 100;

	uint seed;

	//
	for (uint set=0; set < train->sets; set++) {
		seed = (uint)pseudo_randomi(start_seed + set*inst);

		kconvl33samepool22max_train_const_MemCpyToSymbol(train->_weight + set*(mdl->weights)+wstart, n0*n1*9);
		
		kconvl33samepool22max_backward_const_th1x1<<<dim3(KERN_DIV(Ax/2,16), KERN_DIV(Ay/2,16), n1), dim3(16,16,1)>>>(
			n0, n1, Ax, Ay,
			activ,
			time,
			mdl->total, mdl->weights, mdl->locds,
			istart, wstart, ystart, locdstart,
			train->_var, train->_weight, train->_locd,
			train->_grad, train->_meand,
			seed, drop_rate,
			set, train->sets);

		cudaDeviceSynchronize();
	}
};