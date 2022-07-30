'''DOT2D = 1

TEST_MODEL_DOT2D = {	############# DOT2D ##############
	#(31,32) -> (15,32)
	#	y = f(xW + b)     x:(31,32), a:(31,15), b:(15,32), y:(15,32)
	'insts' : [
		(DOT2D, [Ax:=31,Ay:=32,Bx:=15, 0, 0,Ax*Ay,0,0, 0])
	],
	'vars' : Bx*Ay,
	'weights':Ax*Bx + Bx*Ay, 
	'locds':Bx*Ay,
	'inputs':Ax*Ay,
	'outputs':Bx*Ay,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',Ax*Ay)],
	'wsep': [('W',0),('B',Ax*Bx)],
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

class TEST_MODEL_DOT2D(TEST_MDL):
	mdl = MDL(
		[
			DOT2D([Ax:=31,Ay:=32,Bx:=15, 0, 0,Ax*Ay,0,0, 0])
	
		],
		inputs:=Ax*Ay,
		outputs:=Bx*Ay,
		_vars:=Bx*Ay,
		w:=[random() for _ in range(Ax*Bx + Bx*Ay)],
		locds:=Bx*Ay
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',Ax*Ay)]
	wsep = [('W',0),('B',Ax*Bx)]
	lsep = 	[('y locd',0)]