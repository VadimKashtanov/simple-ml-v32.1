#include "pkg_head/insts/lstm1d.cuh"

//			   0  1    2      3      4       5          6 
//Arguments = [X, Y, istart,ystart,wstart,locdstart, drop_rate]

__global__
void lstm1d_use_th1x1(
	uint X, uint Y,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	/*   
			<<< grid==dim3(Y)  block==dim3(1) >>>
	*/

	uint y = threadIdx.x + blockIdx.x * blockDim.x;

	if (y < Y) {

		uint inp = total*line + istart;
		uint W = wstart;

		uint lineW = X*Y + Y*Y + Y;	// == sizeof(W + U + B). There is 4 sets of (W,U,B) for f0,f1,f2 and g0

		uint vpos, wpos;

		float f0=0, f1=0, f2=0, g0=0;

		float tmpt;

		//	x @ W
		for (uint k=0; k < X; k++) {
			vpos = inp + k;
			wpos = k*X + y;

			tmpt = var[vpos];
			f0 += tmpt * weight[W + wpos];
			f1 += tmpt * weight[W + lineW + wpos];
			f2 += tmpt * weight[W + 2*lineW + wpos];
			g0 += tmpt * weight[W + 3*lineW + wpos];
		}

		//	h[-1] @ U
		if (time > 0) {
			for (uint k=0; k < X; k++) {
				vpos = (time-1)*total + istart + Y + k; 		//out - total == total*(l-1) + ystart
				wpos = X*Y + k*X + y;

				tmpt = var[vpos];
				f0 += tmpt * weight[W + wpos];
				f1 += tmpt * weight[W + lineW + wpos];
				f2 += tmpt * weight[W + 2*lineW + wpos];
				g0 += tmpt * weight[W + 3*lineW + wpos];
			}
		}

		//	+ B
		wpos = X*Y + Y*Y + y;
		f0 += weight[W + wpos];
		f1 += weight[W + lineW + wpos];
		f2 += weight[W + 2*lineW + wpos];
		g0 += weight[W + 3*lineW + wpos];

		f0 = logistic(f0);
		f1 = logistic(f1);
		f2 = logistic(f2);
		g0 = tanh(g0);

		if (time > 0) e_1 = var[(time-1)*total + ystart + y];
		else e_1 = 0; 
		
		float e = f0*e_1 + f1*g0;
		float h = f2 * e;

		var[time*total + ystart + y] = e;
		var[time*total + ystart + Y + y] = h;
	}
};