#include "pkg_head/insts/lstm1d.cuh"

__global__
void lstm1d_forward_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	//	<<<Y,sets>>>
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (y < Y) {
		uint inp = total*sets*time + total*set + istart;
		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = X*Y;
		uint _U = Y*Y;
		uint _B = Y;

		uint lineW = _W + _U + _B;

		uint vpos, wpos;

		// f0,f1,f2 = logistic(x@W + h[-1]@U + B)
		// g0 	  = tanh 	(x@W + h[-1]@U + B)
		float f0=0,f1=0,f2=0,g0=0;

		float tmpt;

		// .W
		for (uint k=0; k < X; k++) {	//for all in INPUT
			//	Positions
			vpos = inp + k;

			//	Drop
			if (pseudo_randomf(seed + vpos) > drop_rate) {	//pas de drop sur h[-1]@.U car h[-1] n'est pas un input mais un output et h[-1] aura deja l'influence
				
				//
				wpos = k*Y + x;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + 0*lineW + wpos];
				f1 += tmpt * weight[W + 1*lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		// .U
		if (time > 0) {
			for (uint k=0; k < Y; k++) {
				vpos = total*sets*(total-1) + total*set + ystart + Y + k;
				wpos = _W + k*Y + x;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + 0*lineW + wpos];
				f1 += tmpt * weight[W + 1*lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		// .B
		wpos = _W + _U + y;
		f0 += w[W + 0*lineW + wpos];
		f1 += w[W + 1*lineW + wpos];
		f2 += w[W + 2*lineW + wpos];
		g0 += w[W + 3*lineW + wpos];

		// activ(_sum)
		f0 = logistic(f0);
		f1 = logistic(f1);
		f2 = logistic(f2);
		g0 = tanh(g0);

		// e = f0 * e[-1] + f1 * g0
		// l - 1 have to be >= 0
		float e_1;
		if (time > 0) e_1 = var[total*sets*(time-1) + total*set + ystart + y];
		else e_t = 0;
		
		float e = f0*e_1 + f1*g0;
		float h = f2 * e;

		locd[locdpos + 0*Y + y] = f0;//f2*e_1*( f0*(1 - f0) );	//	f0 locd
		locd[locdpos + 1*Y + y] = f1;//f2*g0*( f1*(1 - f1) );	//	f1 locd
		locd[locdpos + 2*Y + y] = f2;//e*( f2*(1 - f2) );		//	f2 locd
		lcod[locdpos + 3*Y + y] = g0;//f2*f1*( 1 - g0*g0);		//	g0 locd

		var[out + 0*Y + y] = e;
		var[out + 1*Y + y] = h;
	}
};