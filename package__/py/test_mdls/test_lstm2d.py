'''LSTM2D = 5 						#[Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]

TEST_MODEL_LSTM2D = {	############# LSTM2D ##############
	#(6) -> (8)
	#	y = lstm(x)
	'insts' : [
		(LSTM2D, [Ax:=6,Ay:=7,Bx:=8, 0,Ax*Ay,0,0, 0])
	],

	'vars' : 2*Bx*Ay,
	'weights': 4*(wline:=(Bx*Ax + Bx*Bx + Bx*Ay)),
	'locds': 4*Bx*Ay,
	'inputs': Ax*Ay,
	'outputs': 2*Bx*Ay,
	'lines':2,
	'sets':3,

	'vsep': [('input',0), ('e',Ax*Ay), ('h',Ax*Ay+Bx*Ay)],
	'wsep': [
		('Wf0',0),('Uf0',Bx*Ax),('Bf0',Bx*Ax+Bx*Bx),
		('Wf1',wline),('Uf1',wline+Bx*Ax),('Bf1',wline+Bx*Ax+Bx*Bx),
		('Wf2',2*wline),('Uf2',2*wline+Bx*Ax),('Bf2',2*wline+Bx*Ax+Bx*Bx),
		('Wg0',3*wline),('Ug0',3*wline+Bx*Ax),('Bg0',3*wline+Bx*Ax+Bx*Bx)
	],
	'lsep': [('locd.f0',0),('locd.f1',Bx*Ay),('locd.f2',2*Bx*Ay),('locd.g0',3*Bx*Ay)],
	
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

class TEST_MODEL_LSTM2D(TEST_MDL):
	mdl = MDL(
		[
			LSTM2D([Ax:=6,Ay:=7,Bx:=8, 0,Ax*Ay,0,0, 0])
	
		],
		inputs:=Ax*Ay,
		outputs:=2*Bx*Ay,
		_vars:=2*Bx*Ay,
		w:=[random() for _ in range(4*(wline=(Bx*Ax + Bx*Bx + Bx*Ay)))],
		locds:=4*Bx*Ay
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('e',Ax*Ay), ('h',Ax*Ay+Bx*Ay)]
	wsep = [
		('Wf0',0),('Uf0',Bx*Ax),('Bf0',Bx*Ax+Bx*Bx),
		('Wf1',wline),('Uf1',wline+Bx*Ax),('Bf1',wline+Bx*Ax+Bx*Bx),
		('Wf2',2*wline),('Uf2',2*wline+Bx*Ax),('Bf2',2*wline+Bx*Ax+Bx*Bx),
		('Wg0',3*wline),('Ug0',3*wline+Bx*Ax),('Bg0',3*wline+Bx*Ax+Bx*Bx),
	]
	lsep = [('locd.f0',0),('locd.f1',Bx*Ay),('locd.f2',2*Bx*Ay),('locd.g0',3*Bx*Ay)]
	