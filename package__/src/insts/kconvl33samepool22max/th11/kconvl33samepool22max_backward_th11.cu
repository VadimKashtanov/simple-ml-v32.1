#include "pkg_head/insts/kconvl33samepool22max.cuh"

__global__
void kconvl33samepool22max_backward_const_th1x1(
	uint n0, uint n1, uint Ax, uint Ay,
	uint activ,
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	float * var, float * weight, float * locd,
	float * grad, float * meand,
	uint seed, float drop_rate,
	uint set, uint sets)
{
	/*
		On divise la grille avec 1 kernel par pixel.
		Vu que chaqu'un des pixels fait pool22max, il y un block avec 4 _xx de convolution.
		la derivee local de y = max(a,b,c,d) en a,b,c,d est null pour ceux qui ne sont pas max.
		Donc au finale on calcule que pour un seul _xx (ce pixel est dans `.k = .x | .K`).
		A partire de la on peut considere que max existe plus, vu que c'est pour lui la fonction (lambda x:x), donc c'est juste une copie.
		Donc la y = activation(_xx + bias, activ). Donc dL/d_xx += dL/dY * dY/d(activation) * d(activation)/d_xx
		Un peut comme dans dot2d on ajoute a d(pixel maximum):
			dL/dY 				c'est juste l'erreur, qui sera dans un gros model, juste le gradient de l'input de l'instruction suivante
			dY/d(activation) 	c'est la dérivée local : locd[0]  (ou locd[1] est le max_id), chaque pixel de Y a 2 locd
			d(activation)/d_xx 	c'est ducoup le kernel (vu que c'est une simple multiplication)

		la derivee du bias c'est just dL/d_xx += dL/dY * dY/d(activation)
		car d(activation)/dbias == 1  (y = x + 1*b => dy/db = 1)

		Ici l'algorithm fait juste un ajustement avec (y,x) et (pool_block_x,pool_block_y) pour se mettre sur le bon pixel maximum
	*/
	uint out_x = threadIdx.x + blockIdx.x*blockDim.x,	\	//+1 because we don't compute border of output
		 out_y = threadIdx.y + blockIdx.y*blockDim.y;		//+1 it's an usefull approximation
	uint _n1   = threadIdx.z + blockIdx.z*blockDim.z;

	if (out_x < Ax/2 && out_y < Ay/2 && _n1 < n1) {
		uint _Ax = out_x * 2;
		uint _Ay = out_y * 2;

		//	Debut du kernel et de l'image
		uint kstart = _n1*9*n0;
		uint K = set*wsize + wstart;
		uint istart = time*sets*total + set*total + istart;

		//	On load les local derivee et le pixel maximum
		uint this_y_pixel_locd = time*sets*lsize + set*lsize + lstart + _n1*2*(Ax*Ay/4) + out_y*2*(Ax/2) + 2*out_x;
		float __locd = 	locd[this_y_pixel_locd];
		uint max_id = (uint)locd[this_y_pixel_locd + 1];

		//	Erreur * derivee local
		float dLdS = grad[time*sets*total + set*total + ystart + _n1*(Ax*Ay/4) + out_y*(Ax/2) + out_x] * __locd;
		
		uint imgpos;
		int y,x;

		int pool_block_x=max_id%2, pool_block_y=(max_id - max_id%2)/2;

		uint bias = set*wsize + wstart + n0*n1*9 + _n1*Ax*Ay + _Ay*Ax + _Ax;
		meand[bias + pool_block_y*Ax + pool_block_x] += dLdS;

		for (uint _n0=0; _n0 < n0; _n0++) {
			for (int i=-1; i < 2; i++) {
				for (int j=-1; j < 2; j++) {
					y = _Ay + i + pool_block_y;
					x = _Ax + j + pool_block_x;

					imgpos = istart + _n0*Ax*Ay + y*Ax + x;

					if (pseudo_randomf(imgpos + seed) >= drop_rate && y >= 0 && x >= 0 && y < Ay && x < Ax) {
						atomicAdd(&grad[imgpos], 		dLdS * const_mem[kstart]);
						atomicAdd(&meand[K + kstart], 	dLdS * var[imgpos]);
					}

					kstart++;
				}

			}
		}
	}
};