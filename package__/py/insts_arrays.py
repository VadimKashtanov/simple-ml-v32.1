INST_ORDER = [
	"DOT1D",
	"DOT2D",
	"KCONVL33SAMEPOOL22MAX", 
	"SOFTMAX",
	"LSTM1D",
	"LSTM2D",
	"GAUSSFILTRE1D",
	"GAUSSFILTRE2D",
	"DOT1DRECURENT",
	"DOT2DRECURENT",
	"DOTGAUSSFILTRE2D"
]

#DOT1D = 0 						#[Ax,Yx, activ, input_start,ystart,wstart,locdstart, drop_rate]
#DOT2D = 1 						#[Ax,Ay,Bx, activ, input_start,ystart,wstart,locdstart, drop_rate]
#KCONVL33SAMEPOOL22MAX = 2 		#[Ax,Ay, n0,n1, activ, input_start,ystart,wstart,locdstart, drop_rate]
#SOFTMAX = 3 					#[len, input_start, ystart]
#LSTM1D = 4 						#[X,Y, istart,ystart,wstart,locdstart, drate]
#LSTM2D = 5 						#[Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]
#GAUSSFILTRE1D = 6 				#[len, istart,ystart,wstart,lstart]
#GAUSSFILTRE2D = 7  				#[X,Y, istart,ystart,wstart,lstart]
#DOT1DRECURENT = 8  				#[Ax,At, Yx, activ, ist,yst,wst,lst, drate]
#DOT2DRECURENT = 9  				#[Ax,Ay,At,Bx,activ,istart,ystart,wstart,lstart, drate]
#DOTGAUSSFILTRE2D = 10  			#[Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate]

INSTS = []

for inst in INST_ORDER:
	exec(f"from .package.py.insts.{inst.lower()} import {inst}")
	exec(f"{INSTS} += [{inst}]")