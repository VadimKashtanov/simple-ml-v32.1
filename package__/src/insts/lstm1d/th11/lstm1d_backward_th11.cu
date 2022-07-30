#include "pkg_head/insts/lstm1d.cuh"

/*			  *=====*
			  |     |
			  |		|
			  |	.W	|
			  |		|
			  |		|
			  *=====*
*===========* *=====*
|	.input	| | 	|	input@W
*===========* *=====*
				 +
			  *=====*
			  |	.U  |
			  |	    |
			  *=====*
	*=======* *=====*
	| h[-1] | |		|  h[-1]@U
	*=======* *=====*
				 +
			  *=====*
			  |	.B	|
			  *=====*
*/

__global__
void lstm1d_backward_INPUT_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drate,
	uint sets)
{
	/*	Backward grad(input)
		meand(.W) of f0,f1,f2,g0
	*/

	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	uint inp = total*sets*time + total*_set + istart + x;

	if (x < X && pseudo_randomf(seed + inp) > drop_rate) {	//if input[x] is droped, following will be *0

		//
		//	each `x` use all the locd, so an dynamic shared memory could be usefull
		//	save on extern __shared__ dsf0,dsf1,dsf2,dsg0

		float grad_input_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint lineW = X*Y + Y*Y + Y;

		uint vpos = total*sets*time + set*total + istart + x;
		float xval = var[vpos];

		float chain_deriv;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos;

		//	Backward W
		for (uint k=0; k < Y; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y

			epos = out + k;
			e_1pos = total*sets*(time-1) + total*_set + ystart + k; //if l == 0 , e_1pos <= 0
			hpos = out + Y + k;

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Y + k];// * dH;
			f1 = locd[locdpos + 1*Y + k];// * dH;
			f2 = locd[locdpos + 2*Y + k];// * dH;
			g0 = lcod[locdpos + 3*Y + k];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			//if time > 0:
			grad[e_1pos] += de * f0;

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = ws*set + wstart + 0*lineW + (x*Y + k);					//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf0 * xvalue;
			atomicAdd(meand + wpos, dsf0 * xvalue);
			grad_input_compute += dsf0 * weight[wpos];

			//	f1
			wpos = ws*set + wstart + 1*lineW + (x*Y + k);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf1 * xvalue;
			atomicAdd(meand + wpos, dsf1 * xvalue);
			grad_input_compute += dsf1 * weight[wpos];

			//	f2
			wpos = ws*set + wstart + 2*lineW + (x*Y + k);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf2 * xvalue;
			atomicAdd(meand + wpos, dsf2 * xvalue);
			grad_input_compute += dsf2 * weight[wpos];

			//	g0
			wpos = ws*set + wstart + 3*lineW + (x*Y + k);			//on met a jour que .W pas .U no .B
			//meand[wpos] += dsf2 * xvalue;
			atomicAdd(meand + wpos, dsg0 * xvalue);
			grad_input_compute += dsf2 * weight[wpos];
		}

		//	Backward input
		grad[vpos] += grad_input_compute;
		//atomicAdd(grad + vpos, grad_input_compute);
	}
}

__global__
void lstm1d_backward_H1_BIAS_th1x1(
	uint X, uint Y,
	uint time,
	uint input_start, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsizeize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (y < Y) {	//Only input is under drop. h is an output. It's values, could be droped, but in an other instruction

		float grad_H1_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		//time > 0 because lstm1d.cu call this kernel only on time > 0
		uint h1pos = total*sets*(time-1) + set*total + istart + Y + y;
		float h1val = var[h1pos];

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = X*Y;
		uint _U = Y*Y;
		uint _B = Y;

		uint lineW = _W + _U + _B;
		
		float chain_deriv;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos;
		for (uint k=0; k < Y; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y
			epos = out + k;
			e_1pos = total*sets*(time-1) + total*_set + ystart + k; //if l == 0 , e_1pos <= 0
			hpos = out + Y + k;

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Y + k];// * dH;
			f1 = locd[locdpos + 1*Y + k];// * dH;
			f2 = locd[locdpos + 2*Y + k];//* dH;
			g0 = lcod[locdpos + 3*Y + k];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			//if time > 0:
			grad[e_1pos] += de * f0;

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = ws*set + wstart + 0*lineW + _W + (x*Y + k);					//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf0 * h1val;
			atomicAdd(meand + wpos, dsf0 * h1val);
			grad_H1_compute += dsf0 * weight[wpos];

			//	f1
			wpos = ws*set + wstart + 1*lineW + _W + (x*Y + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf1 * h1val;
			atomicAdd(meand + wpos, dsf1 * h1val);
			grad_H1_compute += dsf1 * weight[wpos];

			//	f2
			wpos = ws*set + wstart + 2*lineW + _W + (x*Y + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf2 * h1val;
			atomicAdd(meand + wpos, dsf2 * h1val);
			grad_H1_compute += dsf2 * weight[wpos];

			//	g0
			wpos = ws*set + wstart + 3*lineW + _W + (x*Y + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsg0 * h1val;
			atomicAdd(meand + wpos, dsg0 * h1val);
			grad_H1_compute += dsg0 * weight[wpos];
		}

		//	Backward h[-1]
		grad[h1pos] += grad_H1_compute;
		//atomicAdd(grad + h1pos, grad_H1_compute);

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Y>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//
		
		epos = out + y;
		e_1pos = total*sets*(time-1) + total*_set + ystart + y; //if l == 0 , e_1pos <= 0
		hpos = out + Y + y;

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Y + y];// * dH;
		f1 = locd[locdpos + 1*Y + y];// * dH;
		f2 = locd[locdpos + 2*Y + y];// * dH;
		g0 = lcod[locdpos + 3*Y + y];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[ws*set + wstart + 0*lineW + _W + _U + (x*Y + k)] += dsf0;

		//	f1
		meand[ws*set + wstart + 1*lineW + _W + _U + (x*Y + k)] += dsf1;

		//	f2
		meand[ws*set + wstart + 2*lineW + _W + _U + (x*Y + k)] += dsf2;

		//	g0
		meand[ws*set + wstart + 3*lineW + _W + _U + (x*Y + k)] += dsg0;
	}
};

__global__
void lstm1d_backward_BIAS_ONLY_th1x1(
	uint X, uint Y,
	uint time,
	uint input_start, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsizeize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (y < Y) {	//Only input is under drop, .h is an output. It's values, could be droped, but in an other instruction

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = X*Y;
		uint _U = Y*Y;
		uint _B = Y;

		uint lineW = _W + _U + _B;

		float chain_deriv;
		float _grad;	//of h[t]
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos;

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Y>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//
		
		epos = out + y;
		e_1pos = total*sets*(time-1) + total*_set + ystart + y; //if l == 0 , e_1pos <= 0
		hpos = out + Y + y;

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Y + y];// * dH;
		f1 = locd[locdpos + 1*Y + y];// * dH;
		f2 = locd[locdpos + 2*Y + y];// * dH;
		g0 = lcod[locdpos + 3*Y + y];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[ws*set + wstart + 0*lineW + _W + _U + (x*Y + k)] += dsf0;

		//	f1
		meand[ws*set + wstart + 1*lineW + _W + _U + (x*Y + k)] += dsf1;

		//	f2
		meand[ws*set + wstart + 2*lineW + _W + _U + (x*Y + k)] += dsf2;

		//	g0
		meand[ws*set + wstart + 3*lineW + _W + _U + (x*Y + k)] += dsg0;
	}
};