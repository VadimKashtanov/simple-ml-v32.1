'''GAUSSFILTRE1D = 6 				#[len, istart,ystart,wstart,lstart]

TEST_MODEL_GAUSSFILTRE1D = {	############# GAUSSFILTRE1D ##############
	'insts' : [
		(GAUSSFILTRE1D, [(_len:=17), 0,_len,0,0])
	],
	'vars' : _len,
	'weights': _len, 
	'locds': _len,
	'inputs': _len,
	'outputs': _len,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('output',_len)],
	'wsep': [('P',0)],
	'lsep':	[('locd',0)],

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

class TEST_MODEL_GAUSSFILTRE1D(TEST_MDL):
	mdl = MDL(
		[
			GAUSSFILTRE1D([(_len:=17), 0,_len,0,0])
	
		],
		inputs:=_len,
		outputs:=_len,
		_vars:=_len,
		w:=[random() for _ in range(_len)],
		locds:=_len
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',_len)]
	wsep = [('P',0)]
	lsep = 	[('locd',0)]