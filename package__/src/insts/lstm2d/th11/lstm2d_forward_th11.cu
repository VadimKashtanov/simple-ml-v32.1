#include "pkg_head/insts/lstm2d.cuh"

__global__
void lstm2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;
	
	if (x < Bx && y < Ay)
	{
		uint inp = total*sets*time + total*set + istart;
		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + locdstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		uint vpos, wpos;

		// f0,f1,f2 = logistic(x@W + h[-1]@U + B)
		// g0 	  = tanh 	(x@W + h[-1]@U + B)
		float f0=0,f1=0,f2=0,g0=0;

		float tmpt;

		// .W
		for (uint k=0; k < Ax; k++) {	//for all in INPUT
			//	Positions
			vpos = inp + (y*Ax + k);

			//	Drop
			if (pseudo_randomf(seed + vpos) > drop_rate) {
				
				//
				wpos = k*Bx + x;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + 0*lineW + wpos];
				f1 += tmpt * weight[W + 1*lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		// .U
		if (time > 0) {
			for (uint k=0; k < Bx; k++) {
				vpos = total*sets*(time-1) + total*set + ystart + (Bx*Ay) + y*Bx + k;	///h[-1]
				wpos = _W + k*Bx + x;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + 0*lineW + wpos];
				f1 += tmpt * weight[W + 1*lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		// .B
		wpos = _W + _U + y*Bx + x;
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
		if (time > 0) e_1 = var[total*sets*(time-1) + total*set + ystart + (y*Bx + x)];
		else e_t = 0;
		
		float e = f0*e_1 + f1*g0;
		float h = f2 * e;

		//	n*Bx*Ay car on stoque 4x la matrice Bx*Ay des derives locales pour la derivee en chaine de f0, f1, f2 et g0
		locd[locdpos + 0*Bx*Ay + (y*Bx + x)] = f0;//f2*e_1*( f0*(1 - f0) );		//	f0
		locd[locdpos + 1*Bx*Ay + (y*Bx + x)] = f1;//f2*g0*( f1*(1 - f1) );	//	f1
		locd[locdpos + 2*Bx*Ay + (y*Bx + x)] = f2;//e*( f2*(1 - f2) );	//	f2
		locd[locdpos + 3*Bx*Ay + (y*Bx + x)] = g0;//f2*f1*( 1 - g0*g0);	//	g0

		//	On stoque dans le output 2 matrices Bx*Ay  ou il y a `e` et `h`.
		//	`h` est le resultat du LSTM
		//	`e` est juste utilise pour avoire la ligne d'apres le e[-1]
		var[out + 0*Bx*Ay + (y*Bx + x)] = e;
		var[out + 1*Bx*Ay + (y*Bx + x)] = h;
	};
};