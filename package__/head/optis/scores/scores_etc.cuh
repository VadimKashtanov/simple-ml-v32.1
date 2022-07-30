#pragma once

#include "kernel/head/opti.cuh"

__global__ void opti_kernel_sum_scores_over_lines(
	float * grad, float * var, float * output,
	uint total, uint lines, uint sets, uint ostart, uint outs);

__global__ void opti_kernel_sum_scores_over_outputs(
	float * grad, float * scores,
	uint total, uint sets, uint ostart, uint outs);