#include "pkg_head/insts/gaussfiltre2d/gaussfiltre2d_th11.cuh"

void gaussfiltre2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint X=mdl->param[inst][0],		\
		 Y=mdl->param[inst][1],		\
		 istart=mdl->param[inst][2],\
		 ystart=mdl->param[inst][3],\
		 wstart=mdl->param[inst][4];

	gaussfiltre2d_use_th1x1<<<dim3(KERN_DIV(len,16)),dim3(16)>>>(
		X, Y,
		time,
		total,
		istart, ystart, wstart,
		var, weight);
}

//======================== Train_t =======================

//-------------------------- forward ---------------------

void gaussfiltre2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 locdstart=mdl->param[inst][5];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	gaussfiltre2d_forward_th1x1<<<dim3(KERN_DIV(len,16), train->sets),dim3(16,1)>>>(
		X, Y,
		time,
		istart, ystart, wstart, lstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd,
		start_seed*inst,
		train->sets);
}

//-------------------------- backward ---------------------

void gaussfiltre2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 locdstart=mdl->param[inst][5];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	gaussfiltre2d_backward_th1x1<<<dim3(KERN_DIV(len,16), train->sets),dim3(16,1)>>>(
		X, Y,
		time,
		istart, ystart, wstart, lstart,
		total, wsize, lsize,
		train->_var, train->_weight, locd, grad, meand,
		start_seed*inst,
		train->sets);
}