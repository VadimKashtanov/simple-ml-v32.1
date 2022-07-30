#include "pkg_head/insts/lstm2d.cuh"

/*			  =======
			  |     |
			  |		|
			  |	.W	|
			  |		|
			  |		|
			  =======
============= =======
|	.input	| | 	|	input@W
============= =======
				 +
			  =======
			  |	.U  |
			  |	    |
			  =======
	  ======= =======
	  |h[-1]| |		|  h[-1]@U
	  ======= =======
				 +
			  =======
			  |	.B	|
			  =======
*/

/*	We could use atomicAdd with 1 direct backward function

*/

__global__  //ca veut dire que le kernel est position sur les cooredonne de l'input, et chaque kernel est associe a un pixel de l'input. Puis on backward on ligne verticale de .W
void lstm2d_backward_INPUT_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Backward grad(input)
		meand(.W) of f0,f1,f2,g0
	*/

	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	uint ipos = total*sets*time + total*set + istart + (y*Ax + x);

	//input = Ax*Ay, and the (x,y) pixel is in input. Then we backward .W and this pixel gradient
	if (x < Ax && y < Ay && pseudo_randomf(seed + ipos) > drop_rate) {	//if input[x] is droped, following will be *0

		float grad_input_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		//uint vpos = total*sets*time + set*total + istart + x;
		float xval = var[ipos];

		float chain_deriv;
		float dH,f0,f1,f2,g0,de;
		float dsf0, dsf1, dsf2, dsg0;

		uint wpos, epos, e_1pos, hpos, outpos;	//wpos   = position du weight en question
												//epos,e_1pos,hpos = output `e` ou `h` (car output = `e` + `h`). e_1 est e[-1]
												//outpos = (y*Bx+k) juste pour calculer de quel pixel de Y nous prenon le locd (car on backward chaque colone de output mais les weights d'une meme ligne) 

		//	Backward W
		for (uint k=0; k < Bx; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y

			outpos = y*Bx + k;

			epos = out + outpos;
			e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
			hpos = out + Bx*Ay + outpos;

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
			f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
			f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
			g0 = lcod[locdpos + 3*Bx*Ay + outpos];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			//if time > 0:
			grad[e_1pos] += de * f0;		//we can't store only 4 locds, because how will we get de*f0 ?

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = ws*set + wstart + 0*lineW + (x*Y + k);			//on met a jour que .W pas .U no .B
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
			//meand[wpos] += dsg0 * xvalue;
			atomicAdd(meand + wpos, dsg0 * xvalue);
			grad_input_compute += dsg0 * weight[wpos];
		}

		//	Backward input
		grad[ipos] += grad_input_compute;
		//atomicAdd(grad + vpos, grad_input_compute);
	}
}

__global__
void lstm2d_backward_H1_BIAS_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint input_start, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint x = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (x < Bx && y < Ay) {	//Only input is under drop. h is an output. It's values, could be droped, but in an other instruction

		float grad_H1_compute = 0;	//_INPUT_ et _H1_ ajoutent un gradient a input[x]

		uint h1pos = total*sets*(time-1) + set*total + istart + (y*Bx + x);	//h[-1] pos
		float h1val = var[h1pos];

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		float chain_deriv;
		float _grad;	//of h[t]

		uint wpos;

		for (uint k=0; k < Bx; k++) {	//[ w0 w1 w2 w3 ... wn]	une ligne du .W (la premiere par exemple)
										//car inp[x] est multiplice par `w[x*Bx + k] for k in Bx`  ou Bx==Y
										//en fait k est la colone de la matrice. la ligne est `y` du kernel
										//et le `x` du kernel determine le pixel `h[-1]` et la ligne dans .U 

			outpos = y*Bx + k;
			
			epos = out + outpos;
			e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
			hpos = out + Bx*Ay + outpos;

			dH = grad[hpos];

			f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
			f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
			f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
			g0 = lcod[locdpos + 3*Bx*Ay + outpos];// * dH;

			de = grad[epos] + dH * f2;	//grad(e) += dH*f2

			grad[epos] = de;

			//if time > 0:
			grad[e_1pos] += de * f0;

			dsf0 = de * var[e_1pos] * f0 * (1 - f0);
			dsf1 = de * g0 * f1 * (1 - f1);
			dsf2 = dH * e * f2 * (1 - f2);
			dsg0 = de * f1 * (1 - g0*g0);

			//	f0
			wpos = W + 0*lineW + _W + (x*Bx + k);					//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf0 * h1val;
			atomicAdd(meand + wpos, dsf0 * h1val);
			grad_h1_compute += dsf0 * weight[wpos];

			//	f1
			wpos = W + 1*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf1 * h1val;
			atomicAdd(meand + wpos, dsf1 * h1val);
			grad_h1_compute += dsf1 * weight[wpos];

			//	f2
			wpos = W + 2*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsf2 * h1val;
			atomicAdd(meand + wpos, dsf2 * h1val);
			grad_h1_compute += dsf2 * weight[wpos];
		
			//	g0
			wpos = W + 3*lineW + _W + (x*Bx + k);			//on met a jour que .U pas .W no .B
			//meand[wpos] += dsg0 * h1val;
			atomicAdd(meand + wpos, dsg0 * h1val);
			grad_h1_compute += dsg0 * weight[wpos];
		}

		//	Backward h[-1]
		grad[h1pos] += grad_input_compute;
		//atomicAdd(grad + vpos, grad_input_compute);

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Bx,Ay>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//

		outpos = y*Bx + x;

		epos = out + outpos;
		e_1pos = total*sets*(time-1) + total*set + ystart + outpos; //if l == 0 , e_1pos <= 0
		hpos = out + Bx*Ay + outpos;

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
		f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
		f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
		g0 = lcod[locdpos + 3*Bx*Ay + outpos];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[W + 0*lineW + _W + _U + (x*Y + k)] += dsf0;

		//	f1
		meand[W + 1*lineW + _W + _U + (x*Y + k)] += dsf1;

		//	f2
		meand[W + 2*lineW + _W + _U + (x*Y + k)] += dsf2;

		//	g0
		meand[W + 3*lineW + _W + _U + (x*Y + k)] += dsg0;
	}
};

__global__
void lstm2d_backward_BIAS_ONLY_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	//
	//	h[-1] @ .U
	//

	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint x = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	if (x < Bx && y < Ay) {	//Only input is under drop. h is an output. It's values, could be droped, but in an other instruction

		uint W = ws*set + wstart;
		uint out = total*sets*time + total*set + ystart;
		uint locdpos = lsize*sets*time + lsize*set + lstart;

		uint _W = Bx * Ax;
		uint _U = Bx * Bx;
		uint _B = Bx * Ay;

		uint lineW = _W + _U + _B;

		float chain_deriv;
		float _grad;	//of h[t]

		//  ============================================
		//	Backward .B
		//	Vu que la grille est de <<<Bx,Ay>>> on en profite car .B l'est aussi
		//	Au lieu de cree un autre fonction qui compute le gradient de .B, on le fait directe ici.	
		//
		
		outpos = y*Bx + x;

		epos = out + outpos;
		e_1pos = total*sets*(time-1) + total*_set + ystart + outpos; //if l == 0 , e_1pos <= 0
		hpos = out + Bx*Ay + outpos;

		dH = grad[hpos];

		f0 = locd[locdpos + 0*Bx*Ay + outpos];// * dH;
		f1 = locd[locdpos + 1*Bx*Ay + outpos];// * dH;
		f2 = locd[locdpos + 2*Bx*Ay + outpos];// * dH;
		g0 = lcod[locdpos + 3*Bx*Ay + outpos];// * dH;

		de = grad[epos] + dH * f2;	//grad(e) += dH*f2
		grad[epos] = de;

		//if time > 0:
		grad[e_1pos] += de * f0;

		dsf0 = de * var[e_1pos] * f0 * (1 - f0);
		dsf1 = de * g0 * f1 * (1 - f1);
		dsf2 = dH * e * f2 * (1 - f2);
		dsg0 = de * f1 * (1 - g0*g0);

		//	f0
		meand[W + 0*lineW + _W + _U + (x*Y + k)] += dsf0;

		//	f1
		meand[W + 1*lineW + _W + _U + (x*Y + k)] += dsf1;

		//	f2
		meand[W + 2*lineW + _W + _U + (x*Y + k)] += dsf2;

		//	g0
		meand[W + 3*lineW + _W + _U + (x*Y + k)] += dsg0;
	}
};
