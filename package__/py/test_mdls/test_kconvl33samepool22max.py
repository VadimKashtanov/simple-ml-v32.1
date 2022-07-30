'''KCONVL33SAMEPOOL22MAX = 2

TEST_MODEL_KCONVL33SAMEPOOL22MAX = {	############# KCONVL33SAMEPOOL22MAX ##############
	#(14,18,1) -> (7,9,3)
	#	y = pool(f(x|K + b))     x:(14,18,1), K:(3,3,1,3), b:(14,18,3), y:(7,9,3)
	'insts' : [
		(KCONVL33SAMEPOOL22MAX, [X:=14,Y:=18, n0:=1,n1:=3, 0, 0,X*Y,0,0, 0])
	],

	'vars' : X/2*Y/2*n1,
	'weights': 9*n0*n1 + X*Y*n1, 
	'locds': 2 * X/2*Y/2*n1,
	'inputs': X*Y*n0,
	'outputs': X/2*Y/2*n1,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',X*Y*n0)],
	'wsep': [('K',0),('B',3*3*n0*n1)],
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

class TEST_MODEL_KCONVL33SAMEPOOL22MAX(TEST_MDL):
	mdl = MDL(
		[
			KCONVL33SAMEPOOL22MAX([X:=14,Y:=18, n0:=1,n1:=3, 0, 0,X*Y,0,0, 0])
	
		],
		inputs:=X*Y*n0,
		outputs:=X/2*Y/2*n1,
		_vars:=X/2*Y/2*n1,
		w:=[random() for _ in range(9*n0*n1 + X*Y*n1)],
		locds:=2 * X/2*Y/2*n1
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',X*Y*n0)]
	wsep = [('K',0),('B',3*3*n0*n1)]
	lsep = 	[('y locd',0)]