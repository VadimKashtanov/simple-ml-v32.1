'''GAUSSFILTRE2D = 7  				#[X,Y, istart,ystart,wstart]

TEST_MODEL_GAUSSFILTRE2D = {	############# GAUSSFILTRE2D ##############
	'insts' : [
		(GAUSSFILTRE2D, [X:=17,Y:=15, 0,Y*X,0,0])
	],
	'vars' : X*Y,
	'weights': X, 
	'locds': X*Y,
	'inputs': X*Y,
	'outputs': X*Y,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',X*Y)],
	'wsep': [('P',0)],
	'lsep':	[('locds',0)],

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

class TEST_MODEL_GAUSSFILTRE2D(TEST_MDL):
	mdl = MDL(
		[
			GAUSSFILTRE2D([X:=17,Y:=15, 0,Y*X,0,0])
	
		],
		inputs:=X*Y,
		outputs:=X*Y,
		_vars:=X*Y,
		w:=[random() for _ in range(X)],
		locds:=X*Y
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',X*Y)]
	wsep = [('P',0)]
	lsep = 	[('locds',0)]