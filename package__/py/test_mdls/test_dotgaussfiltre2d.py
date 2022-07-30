'''DOTGAUSSFILTRE2D = 10  			#[Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate]

TEST_MODEL_DOTGAUSSFILTRE2D = {	############# DOTGAUSSFILTRE2D ##############
	'insts' : [
		(DOTGAUSSFILTRE2D, [Ax:=31,Ay:=32,At:=1,Bx:=15, 0, 0,Ax*Ay,0,0, 0])
	],
	'vars' : Bx*Ay,
	'weights': 2*Bx*Ax, 
	'locds': (Bx*Ay) * Ax,
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

class TEST_MODEL_DOTGAUSSFILTRE2D(TEST_MDL):
	mdl = MDL(
		[
			DOTGAUSSFILTRE2D([Ax:=31,Ay:=32,At:=1,Bx:=15, 0, 0,Ax*Ay,0,0, 0])
	
		],
		inputs:=Ax*Ay,
		outputs:=Bx*Ay,
		_vars:=Bx*Ay,
		w:=[random() for _ in range(2*Bx*Ax)],
		locds:=(Bx*Ay) * Ax
	)

	lines = 2
	sets = 3

	vsep = [('input',0), ('output',Ax*Ay)]
	wsep = [('W',0),('B',Ax*Bx)]
	lsep = 	[('y locd',0)]