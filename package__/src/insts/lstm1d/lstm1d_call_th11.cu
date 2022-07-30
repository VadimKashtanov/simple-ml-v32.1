#include "pkg_head/insts/lstm1d/lstm1d.cuh"

// =======================================================
// ===================== Use_t ===========================
// =======================================================

void lstm1d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint X=mdl->param[inst][0],		\
		 Y=mdl->param[inst][1],		\
		 istart=mdl->param[inst][2],\
		 ystart=mdl->param[inst][3],\
		 wstart=mdl->param[inst][4];

	//	Only th11 exists
	lstm1d_forward_th1x1<<<dim3(KERN_DIV(Y,16)),dim3(16)>>>(
		X, Y,
		time,
		mdl->total,
		istart, ystart, wstart,
		use->var, use->weight);
};

// =======================================================
// ==================== Forward ==========================
// =======================================================

void lstm1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 locdstart=mdl->param[inst][5],	\
		 drate=mdl->param[inst][6];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	//	Only th11 exists
	lstm1d_forward_th1x1<<<dim3(KERN_DIV(Y,16),sets),dim3(16,1)>>>(
		X, Y,
		time,
		istart, ystart, wstart, locdstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd,
		inst*start_seed, drate,
		train->sets);
};

// =======================================================
// ==================== Backward =========================
// =======================================================

void lstm1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	/*
		We could use __constant__[] and extern__shared__[] to call only one time from vram each locd
	*/
	Mdl_t * mdl = train->mdl;

	uint X=mdl->param[inst][0],			\
		 Y=mdl->param[inst][1],			\
		 istart=mdl->param[inst][2],	\
		 ystart=mdl->param[inst][3],	\
		 wstart=mdl->param[inst][4],	\
		 locdstart=mdl->param[inst][5],	\
		 drate=mdl->param[inst][6];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	//	Only th11 exists
	lstm1d_backward_INPUT_th1x1<<<dim3(KERN_DIV(X,16),sets),dim3(16,1)>>>(		//backward Input (size = X)
		X, Y,
		time,
		istart, ystart, wstart, locdstart,
		total, wsize, locdsize,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drate,
		train->sets);

	if (time == 0) {
		lstm1d_backward_BIAS_ONLY_th1x1<<<dim3(KERN_DIV(Y,16),sets),dim3(16,1)>>>(	//backward output (size = Y)
			X, Y,
			time,
			istart, ystart, wstart, locdstart,
			total, wsize, locdsize,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, drate,
			train->sets);
	} else {
		lstm1d_backward_H1_BIAS_th1x1<<<dim3(KERN_DIV(Y,16),sets),dim3(16,1)>>>(	//backward output (size = Y)
			X, Y,
			time,
			istart, ystart, wstart, locdstart,
			total, wsize, locdsize,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, drate,
			train->sets);
	}
};