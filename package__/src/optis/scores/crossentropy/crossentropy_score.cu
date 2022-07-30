#include "package/head/optis/scores/crossentropy.cuh"

#ifndef ln
	#define ln(x) log(x)
#endif 	// ln = log(x, e) but log(x, e) == log(e)  in C
		//	It's more for the form and clarity than for optimisation 

#define CROSSENTROPY_LOSS(w, g)  w*ln(g) + (1-w)*ln(1 - g)
#define CROSSENTROPY_DLOSS(w, g) (w - g) / (g * (1-g))

//		d(w*ln(g) + (1-w)*ln(1-g))/dg
//	=	w/g - (1-w)/(1-g)
//	=	w(1-g)/(g*(1-g)) - g*(1 - w) / (1 - g)*g
//	= [w*(1-g) - g*(1-w)] / (g*(1 - g))
//	= [w - wg - g + gw] / [g*(1 - g)]
//	= [w - g]/[g * (1 - g)] 

//=================================================================================================
//===================================== dLOSS(g,w)/dg =============================================
//=================================================================================================

static __global__ void opti_kernel_ce_dloss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		grad[pos] = CROSSENTROPY_DLOSS(var[pos], output[line*outs + out]);
	};
};

void CROSSENTROPY_dloss(Opti_t * opti) {
	Train_t * train = opti->train;

	kernel_ce_loss<<<dim3(KERN_DIV(train->mdl->outputs, 16), KERN_DIV(train->data->lines, 16), train->sets),dim3(16, 16, 1)>>>(
		train->_grad, train->_var, train->data->output_d,
		train->mdl->total, train->mdl->vars, train->data->lines, train->data->outputs
	);
};

//=================================================================================================
//====================================== LOSS(g,w) ================================================
//=================================================================================================

static __global__ void opit_kernel_ce_loss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		float g = var[pos];
		float w = output[line*outs + out];
		grad[pos] = CROSSENTROPY_LOSS(g, w);
	};
};

void CROSSENTROPY_score(Opti_t * opti) {
	Train_t * train = opti->train;
	Mdl_t * mdl = train->mdl;

	uint outs = mdl->outputs;
	uint lines = train->data->lines;
	uint sets = train->sets;
	uint out_start = mdl->vars;

	opti_kernel_ce_loss<<<dim3(KERN_DIV(outs, 16), KERN_DIV(lines, 16), sets),dim3(16,16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, out_start, lines, outs);

	opti_kernel_sum_scores_over_lines<<<dim3(KERN_DIV(outs, 16), sets),dim3(16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, lines, sets, out_start, outs);

	opti_kernel_sum_scores_over_outputs<<<dim3(sets),dim3(1)>>>(
		train->_grad, opti->set_score_d,		//	<---- ??
		mdl->total, sets, output_start, outs);
};