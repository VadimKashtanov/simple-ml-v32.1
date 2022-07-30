#include "package/head/optis/scores/meansquared.cuh"

#define MEANSQUARED_LOSS(w, g) pow(g - w, 2)/2
#define MEANSQUARED_DLOSS(w, g)	g - w

//=================================================================================================
//===================================== dLOSS(g,w)/dg =============================================
//=================================================================================================

static __global__ void opti_kernel_ms_dloss(
	float * grad, float * var, float * output,
	uint total, uint ostart, uint lines, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint line = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (out < outs && line < lines)
	{
		uint pos = line*sets*total + set*total + ostart + out;
		grad[pos] = MEANSQUARED_DLOSS(var[pos], output[line*outs + out]);
	};
};

void MEANSQUARED_dloss(Opti_t * opti) {
	Train_t * train = opti->train;

	kernel_ms_loss<<<dim3(KERN_DIV(train->mdl->outputs, 16), KERN_DIV(train->data->lines, 16), train->sets),dim3(16, 16, 1)>>>(
		train->_grad, train->_var, train->data->output_d,
		train->mdl->total, train->mdl->vars, train->data->lines, train->data->outputs
	);
};

//=================================================================================================
//====================================== LOSS(g,w) ================================================
//=================================================================================================

static __global__ void opit_kernel_ms_loss(
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
		grad[pos] = MEANSQUARED_LOSS(g, w);
	};
};

void MEANSQUARED_score(Opti_t * opti) {
	Train_t * train = opti->train;
	Mdl_t * mdl = train->mdl;

	uint outs = mdl->outputs;
	uint lines = train->data->lines;
	uint sets = train->sets;
	uint out_start = mdl->vars;

	opti_kernel_ms_loss<<<dim3(KERN_DIV(outs, 16), KERN_DIV(lines, 16), sets),dim3(16,16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, out_start, lines, outs);

	opti_kernel_sum_scores_over_lines<<<dim3(KERN_DIV(outs, 16), sets),dim3(16,1)>>>(
		train->_grad, train->_var, train->data->output_d,
		mdl->total, lines, sets, out_start, outs);

	opti_kernel_sum_scores_over_outputs<<<dim3(sets),dim3(1)>>>(
		train->_grad, opti->set_score_d,		//	<---- ??
		mdl->total, sets, output_start, outs);
};