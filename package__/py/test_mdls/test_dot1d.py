'''DOT1D = 0 						#[Ax,Yx, activ, input_start,ystart,wstart,locdstart, drop_rate]

TEST_MODEL_DOT1D = {	############# DOT1D ##############
	#(31) -> (15)
	#	y = f(xW + b)     x:(31), W:(15,31), b:(15), y:(15)
	'insts' : [
		(DOT1D, [Ax:=31,Yx:=15, 0, 0,Ax,0,0, 0])
	],

	'vars' : Yx,
	'weights':Ax*Yx + Yx, 
	'locds':Yx,
	'inputs':Ax,
	'outputs':Yx,

	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',31)],
	'wsep': [('W',0),('B',Ax*Yx)],
	'lsep':	[('y locd',0)],

	'optis_args' : [
		[],	#sgd
		[],	#rmsprop
		[],	#adam
	],

	'gtics_args' : [
		['elites=10%'],	#elite
	]
}'''

from .package.py.insts_arrays import INSTS
from random import random

class TEST_MODEL_DOT1D(TEST_MDL):
	mdl = MDL(
		[
			DOT1D([Ax:=31,Yx:=15, 0, 0,Ax,0,0, 0])
	
		],
		inputs:=Ax,
		outputs:=Yx,
		_vars:=Yx,
		w:=[random() for _ in range(Ax*Yx + Yx)],
		locds:=Yx
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',31)]
	wsep = [('W',0),('B',Ax*Yx)]
	lsep = 	[('y locd',0)]


	#ajouter ca partout en copier coller ou alors mettre de maniere generale ces parametres de teste dans les Class SGD, MOMENTUM.... et ELITE et GENETIQUE1
	#																peut etre mieux de mettre un argument de test du style `DELFAUT_ARGS` = {'arg0' : 'insert arg', ...}
	optis_args : [
		[],	#sgd
		[],	#rmsprop
		[],	#adam
	]

	gtics_args : [
		['elites=10%'],	#elite
	]