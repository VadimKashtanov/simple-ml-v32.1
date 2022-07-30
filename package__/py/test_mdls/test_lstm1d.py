'''LSTM1D = 4 						#[X,Y, istart,ystart,wstart,locdstart, drate]

TEST_MODEL_LSTM1D = {	############# LSTM1D ##############
	#(6) -> (8)
	#	y = lstm(x)
	'insts' : [
		(LSTM1D, [X:=6,Y:=8, 0,X,0,0, 0])
	],

	'vars' : 2*Y,
	'weights': 4*(wline:=(X*Y + Y*Y + Y)), 
	'locds': 4*Y,
	'inputs': X,
	'outputs': 2*Y,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('e',X), ('h',X+Y)],
	'wsep': [
		('Wf0',0),('Uf0',X*Y),('Bf0',X*Y+Y*Y),
		('Wf1',wline),('Uf1',wline+X*Y),('Bf1',wline+X*Y+Y*Y),
		('Wf2',2*wline),('Uf2',2*wline+X*Y),('Bf2',2*wline+X*Y+Y*Y),
		('Wg0',3*wline),('Ug0',3*wline+X*Y),('Bg0',3*wline+X*Y+Y*Y),
		],
	'lsep':	[('locd.f0',0),('locd.f1',Y),('locd.f2',2*Y),('locd.g0',3*Y)],
	
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

class TEST_MODEL_LSTM1D(TEST_MDL):
	mdl = MDL(
		[
			LSTM1D([X:=6,Y:=8, 0,X,0,0, 0])
	
		],
		inputs:=X,
		outputs:=2*Y,
		_vars:=2*Y,
		w:=[random() for _ in range(4*(wline=(X*Y + Y*Y + Y)))],
		locds:=4*Y
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('e',X), ('h',X+Y)]
	wsep = [
		('Wf0',0),('Uf0',X*Y),('Bf0',X*Y+Y*Y),
		('Wf1',wline),('Uf1',wline+X*Y),('Bf1',wline+X*Y+Y*Y),
		('Wf2',2*wline),('Uf2',2*wline+X*Y),('Bf2',2*wline+X*Y+Y*Y),
		('Wg0',3*wline),('Ug0',3*wline+X*Y),('Bg0',3*wline+X*Y+Y*Y),
		]
	lsep = 	[('locd.f0',0),('locd.f1',Y),('locd.f2',2*Y),('locd.g0',3*Y)]
	