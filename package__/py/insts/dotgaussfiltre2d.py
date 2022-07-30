from math import exp

'''

Old version used to multily by a weight the a value.

	  	      [p0,p1]
		      [p2,p3]
		      [p4,p5]
[a0,a1,a2] -> [y0,y1]
[a3,a4,a5] -> [y2,y3]

Y[y*Bx + x] = sum( exp(-(a[y*Ax + i] + p[i*Bx + x])^2) for i in range(Ax))

y0 = exp(-(a0+p0)^2) + exp(-(a1+p2)^2) + exp(-(a2+p4)^2)
y3 = exp(-(a3+p1)^2) + exp(-(a4+p3)^2) + exp(-(a5+p5)^2)

locd = -2(a+p)y

Donc ca va faire Ax * (Bx*Ay) locds qui ne vont etre load qu'une seule fois par 1 meme kernel.

Dans ce cas si utilise pas de locd on va devoire load (Bx*Ay) * (2 * Ax) et recalculer la somme des exp(-(a0+p0)^2)

Les comparaison de vitesses et d'optimisation je les ferais plus tard.
La je cherche juste a faire un truc qui prend pas trop de ressources

'''

'''	Var
Y : Bx*Ay
'''

'''	Weights
P : Bx*Ax
'''

'''	Locds
(Bx*Ay) * Ax
'''

class DOTGAUSSFILTRE2D:

	name = "DOTGAUSSFILTRE2D"

	params_names = ['Ax','Ay', 'Bx', 'istart','ystart','wstart','locdstart', 'drate']

	#	Il ne sert a rien de rajouter Ax * Bx*Ay  locds car de toute facon l'accees a la memoire est similairement aussi long dans les 2 cas

	#	Params : [Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate]

	def check(self):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return all(i >= 0 for i in params) and Ax > 0 and Ay > 0 and Bx > 0 and drate <= 100
	
	def mdl(self,
		total:int, line:int,
		var:[float], w:[float]):
	
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		for x in range(Bx):
			for y in range(Ay):
				_sum = 0
	
				for i in range(Ax):
					apos = line*total + istart + y*Ax
					ppos = wstart + i*Bx + x
					
					_sum += exp(-(var[apos] + w[ppos])**2)
	
				var[line*total + ystart + y*Bx + x] = _sum
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		for x in range(Bx):
			for y in range(Ay):
				_sum = 0
	
				for i in range(Ax):
					apos = l*total + istart + y*Ax
					ppos = wstart + i*Bx + x
	
					_sum += exp(-(var[apos] + w[ppos])**2)
	
					locd[line*sets*locds + _set*locds + locdstart + Ax*(y*Bx+x) + i] = -2*(var[apos] + w[ppos])*exp(-(var[apos] + w[ppos])**2)
	
				var[line*sets*total + _set*total + ystart + (y*Bx+x)] = _sum
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		for x in range(Bx):
			for y in range(Ay):
				dy = grad[l*sets*total + _set*total + ystart + (y*Bx+x)]
	
				for i in range(Ax):
					apos = l*total + istart + y*Ax
					ppos = wstart + i*Bx + x
	
					_locd = locd[line*sets*locds + _set*locds + locdstart + Ax*(y*Bx+x) + i]
	
					grad[apos] += _locd * dy
					grad[ppos] += _locd * dy
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ay
	
	def buildstackmodel_weights(self):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ax
	
	def buildstackmodel_locds(self):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return (Bx*Ay) * Ax
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return [(f'{_id}.Y [dotgaussfiltre2d]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return [(f'{_id}.P [dotgaussfiltre2d]',stack_start)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay, Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return [(f'{_id}.Y [dotgaussfiltre2d]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_dotgaussfiltre2d = "Ax", "Ay", "Bx", "drop_rate"
	
	requiredposition_dotgaussfiltre2d = 1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_dotgaussfiltre2d(self, ystart, wstart, lstart, required):
		Ax, Ay, Bx, drate = required
		return Ax,Ay, Bx, istart,ystart,wstart,lstart, drate