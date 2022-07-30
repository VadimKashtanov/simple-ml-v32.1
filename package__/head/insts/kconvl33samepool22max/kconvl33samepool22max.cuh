#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"


#include "pkg_head/insts/kconvl33samepool22max/kconvl33samepool22max_th11.cuh"

/*
	`const` : n0*9*n1 kernel is loaded in __constant__
	_th11 : each thread compute 1 pixel from Y

	[Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate]
*/

/*	Pourquoi les fonction `cudamemcpytosymbole` sont ici ?
Un __constant__ est compilé dans son unité de compilation.
Donc pour que 2 functions utilises le meme element __constant__ il faut les écrire dans le même fichier.
Concretement ici il y a `cudamemcpytosymbole` et `kconvl33samepool22max_use_const_th1x1` qui utilises par exemple
le meme const_mem, donc il seront ecrit dans la meme unité avec la __constant__ en static.

De meme pour forward et backward, et toutes les autres fonctions qui utilisent ces fonction.
*/

/* ======== Gpu computation MODS ============
const_th11:
	each kernel compute completely one output pixel. [No shared] [consts] [No texture]
const_texture_th11:
	1 pixel of output [no shared] [consts] [texture]
*/

//=========================== Sizes ===============================

/*
	inputs = n0*Ax*Ay
	vars = n1*(Ay/2)*(Bx/2)
	weights = 9*n0*n1 + Ax*Ay*n1		#bias sont sur le .k pas sur le .pool, donc ont la taille de l'input et pas du pooled
	locds = 2*n1*(Ay/2)*(Ax/2)
*/

void kconvl33samepool22max_check(uint * param);

//======================= Cpu_t Forward ===========================

void kconvl33samepool22max_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void kconvl33samepool22max_use_call_mode_th11(Use_t * use, uint inst, uint time);

void kconvl33samepool22max_use(Use_t * use, uint inst, uint time);

//========================		Train_t	  =========================

//----------------------------- forward ---------------------------

void kconvl33samepool22max_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void kconvl33samepool22max_forward(Train_t * train, uint inst, uint time, uint start_seed);

//----------------------------- backward ---------------------------

void kconvl33samepool22max_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void kconvl33samepool22max_backward(Train_t * train, uint inst, uint time, uint start_seed);