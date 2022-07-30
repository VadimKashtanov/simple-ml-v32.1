'''SOFTMAX = 3 					#[len, input_start, ystart]

TEST_MODEL_SOFTMAX = {	############# SOFTMAX ##############
	#(6) -> (6)
	#	y = e^(-x) / sum(e^(-x))
	'insts' : [
		(SOFTMAX, [X:=6, 0, X])
	],

	'vars' : X,
	'weights': 0, 
	'locds': 0,
	'inputs': X,
	'outputs': X,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',X)],
	'wsep': [],
	'lsep':	[],

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

class TEST_MODEL_SOFTMAX(TEST_MDL):
	mdl = MDL(
		[
			SOFTMAX([X:=6, 0, X])
	
		],
		inputs:=X,
		outputs:=X,
		_vars:=X,
		w:=[random() for _ in range(0)],
		locds:=0
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',X)]
	wsep = []
	lsep = 	[]