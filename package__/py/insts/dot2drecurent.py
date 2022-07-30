from math import exp, tanh

activate = [
	lambda x: 1 / (1 + exp(-x)),
	tanh,
	lambda x: exp(-x*x),
	lambda x: x * (x >= 0) 
]

localderiv = [
	lambda x: activate[0](x) * (1 - activate[0](x)),
	lambda x: 1 - tanh(x)**2,
	lambda x: -2*x*activate[2](x),
	lambda x: (x >= 0)
]

class DOT2DRECURENT:

	name = "DOT2DRECURENT"

	params_names = ['Ax','Ay','At', 'Bx', 'activ', 'istart','ystart','wstart','lstart', 'drate']

	#	Params : [Ax,Ay,At, Bx, activ, ist,yst,wst,lst, drate]
	#	At - de combien de lignes on va en arriere. Si At=1 =>  A=A[t-1]

	def check(self):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return activ < 4, Ax > 0 and Bx > 0 and Ay > 0 and 100 >= drop_rate >= 0 and all(i>=0 and int(i)==i for i in params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
	
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		for y in range(Ay):
			for x in range(Bx):
				_sum = 0
				if At >= l:
					for i in range(Ax):
						_sum += w[wstart + Bx*i + x] * var[(l-At)*total + input_start + Ax*y + i]
				
				_sum += w[wstart + Bx*Ax + y*Bx + x]
				
				var[l*total + ystart + y*Bx + x] = activate[activ](_sum)
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		for y in range(Ay):
			for x in range(Bx):
				_sum = 0
				if At >= line:
					for i in range(Ax):
						_sum += w[ws*_set + wstart + Bx*i + x] * var[(line-At)*sets*total + _set*total + input_start + Ax*y + i]
				_sum += w[ws*_set + wstart + Bx*Ax + y*Bx + x]
				
				locd[line*sets*locds + _set*locds + locdstart + y*Bx + x] = localderiv[activ](_sum)
				var[line*sets*total + _set*total + ystart + y*Bx + x] = activate[activ](_sum)
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		for y in range(Ay):
			for x in range(Bx):
				
				dlds = locd[line*sets*locds + _set*locds + locdstart + y*Bx + x] * grad[line*sets*total + _set*total + ystart + y*Bx + x]
				
				meand[ws*_set + wstart + Bx*Ax + y*Bx + x] += dlds
	
				if At >= line:
					for i in range(Ax):
						#_sum += w[ws*_set + wstart + Bx*i + x] * var[(line-At)*sets*total + _set*total + input_start + Ax*y + i]
						wpos = ws*_set + wstart + Bx*i + x
						vpos = (line-At)*sets*total + _set*total + input_start + Ax*y + i
	
						meand[wpos] += dlds * var[vpos]
						grad[vpos] += dlds * w[wpos]
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return Bx*Ay
	
	def buildstackmodel_weights(self):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return Bx*Ax + Bx*Ay
	
	def buildstackmodel_locds(self):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return Bx*Ay
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot2drecurent]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.W [dot2drecurent]',stack_start), (f'{_id}.B [dot2drecurent]',stack_start + Ax*Bx)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay,At, Bx, activ, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot2drecurent]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_dot2drecurent = "Ax", "Ay", "At", "Bx", "activ", "drop_rate"
	
	requiredposition_dot2drecurent = 1,1,1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_dot2drecurent(self, ystart, wstart, lstart, required):
		Ax, Ay, At, Bx, activ, drop_rate = required
		return Ax,Ay,At, Bx, activ, istart,ystart,wstart,lstart, drop_ratclass DOT2DRECURENT(Inst):
	_id = 9
	ID = 9

	name='DOT2DRECURENT'
	params_names=['Ax', 'Ay', 'At', 'Bx', 'activ', 'istart', 'ystart', 'wstart', 'lstart', 'drate']
e