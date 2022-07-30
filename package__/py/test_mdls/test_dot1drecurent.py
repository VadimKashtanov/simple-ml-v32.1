'''DOT1DRECURENT = 8  				#[Ax,At, Yx, activ, ist,yst,wst,lst, drate]

TEST_MODEL_DOT1DRECURENT = {	############# DOT1DRECURENT ##############
	#(31,) -> (15,)
	#	y = f(xW + b)     x:(31,), W:(15,31), b:(15,), y:(15,)
	'insts' : [
		(DOT1DRECURENT, [Ax:=31,At:=1, Yx:=15, 0, 0,Ax,0,0, 0])
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

class TEST_MODEL_DOT1DRECURENT(TEST_MDL):
	mdl = MDL(
		[
			DOT1DRECURENT([Ax:=31,At:=1, Yx:=15, 0, 0,Ax,0,0, 0])
	
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