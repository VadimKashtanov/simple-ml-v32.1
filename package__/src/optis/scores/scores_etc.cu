#include "package/head/optis/scores/scores_etc.cuh"

//	====== Sum all score from all lines to each output =============

//		From
//	l=0	[.......,err(o0),err(o1),err(o2),err(o3)]
//	l=1	[.......,err(o0),err(o1),err(o2),err(o3)]
//	l=2	[.......,err(o0),err(o1),err(o2),err(o3)]
//		To
//	l=0 [......., (1/lines)*(err[l0](o0)+err[l1](o0)+err[l2](o0)), (1/lines)*(err[l0](o1)+err[l1](o1)+err[l2](o1)), (1/lines)*(err[l0](o2)+err[l1](o2)+err[l2](o2)), (1/lines)*(err[l0](o3)+err[l1](o3)+err[l2](o3))]
//	l=1	[.......,err(o0),err(o1),err(o2),err(o3)]
//	l=2	[.......,err(o0),err(o1),err(o2),err(o3)]

//
//	Dessiner tout sur papier puis photocopier et mettre dans la documentation
//
//
__global__ void opti_kernel_sum_scores_over_lines(
	float * grad, float * var, float * output,
	uint total, uint lines, uint sets, uint ostart, uint outs)
{
	uint out = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = threadIdx.y;

	if (out < outs)
	{
		uint pos;
		float _sum_of_lines = 0;
		for (uint l=0; l < lines; l++) {
			_sum_of_lines += grad[l*sets*total + set*total + ostart + out];
		}
		grad[0*sets*total + set*total + ostart + out] = _sum_of_lines / lines;
	};
};

//	========= Sum all outputs to each set ===========

//		From
// set=0	l=0 [......., err[l0](o0)+err[l1](o0)+err[l2](o0), err[l0](o1)+err[l1](o1)+err[l2](o1), err[l0](o2)+err[l1](o2)+err[l2](o2), err[l0](o3)+err[l1](o3)+err[l2](o3)]
// set=1	l=0 [......., err[l0](o0)+err[l1](o0)+err[l2](o0), err[l0](o1)+err[l1](o1)+err[l2](o1), err[l0](o2)+err[l1](o2)+err[l2](o2), err[l0](o3)+err[l1](o3)+err[l2](o3)]

//		To
//	set_scores[0] = (1/(outputs))*sum(set=0 l=0 [......., err[l0](o0)+err[l1](o0)+err[l2](o0), err[l0](o1)+err[l1](o1)+err[l2](o1), err[l0](o2)+err[l1](o2)+err[l2](o2), err[l0](o3)+err[l1](o3)+err[l2](o3)])
//	set_scores[1] = (1/(outputs))*sum(set=1 l=0 [......., err[l0](o0)+err[l1](o0)+err[l2](o0), err[l0](o1)+err[l1](o1)+err[l2](o1), err[l0](o2)+err[l1](o2)+err[l2](o2), err[l0](o3)+err[l1](o3)+err[l2](o3)])

__global__ void opti_kernel_sum_scores_over_outputs(
	float * grad, float * scores,
	uint total, uint sets, uint ostart, uint outs)
{
	uint set = threadIdx.x;

	uint start = 0*sets*total + set*total + ostart + 0;
	float _sum_of_outs = 0;
	for (uint o=0; o < outs; o++) {
		_sum_of_outs += grad[start];
		start++;
	}

	scores[set] = _sum_of_lines / outs;
};