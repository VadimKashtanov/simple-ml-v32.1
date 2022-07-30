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

class DOT2D(Inst):
	_id = 1
	ID = 1

	name='DOT2D'
	params_names=['Ax', 'Ay', 'Bx', 'activ', 'input_start', 'ystart', 'wstart', 'locdstart', 'drop_rate']

	def check(self):
		Ax, Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return Ax > 0 and Bx > 0 and Ay > 0 and 100 >= drop_rate >= 0 and all(i>=0 and int(i)==i for i in params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		Ax, Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
	
		for y in range(Ay):
			for x in range(Bx):
				for i in range(Ax):
					var[l*total + ystart + y*Bx + x] += w[wstart + Bx*i + x] * var[l*total + input_start + Ax*y + i]
				var[l*total + ystart + y*Bx + x] += w[wstart + Bx*Ax + y*Bx + x]
				var[l*total + ystart + y*Bx + x] = activate[activ](var[l*total + ystart + y*Bx + x])
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		Ax, Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
	
		for y in range(Ay):
			for x in range(Bx):
				for i in range(Ax):
					var[line*sets*total + _set*total + ystart + y*Bx + x] += w[ws*_set + wstart + Bx*i + x] * var[line*sets*total + _set*total + input_start + Ax*y + i]
				var[line*sets*total + _set*total + ystart + y*Bx + x] += w[ws*_set + wstart + Bx*Ax + y*Bx + x]
				locd[line*sets*locds + _set*locds + locdstart + y*Bx + x] = localderiv[activ](var[line*sets*total + _set*total + ystart + y*Bx + x])
				var[line*sets*total + _set*total + ystart + y*Bx + x] = activate[activ](var[line*sets*total + _set*total + ystart + y*Bx + x])
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		Ax, Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		
		for y in range(Ay):
			for x in range(Bx):
				dlds = locd[line*sets*locds + _set*locds + locdstart + y*Bx + x] * grad[line*sets*total + _set*total + ystart + y*Bx + x]
				meand[ws*_set + wstart + Bx*Ax + y*Bx + x] += dlds
				for i in range(Ax):
					meand[ws*_set + wstart + Bx*i + x] += dlds * var[line*sets*total + _set*total + input_start + Ax*y + i]
					grad[line*sets*total + _set*total + input_start + Ax*y + i] += dlds * w[ws*_set + wstart + Bx*i + x]
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay, Bx, activ, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ay
	
	def buildstackmodel_weights(self):
		Ax,Ay, Bx, activ, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ax + Bx*Ay
	
	def buildstackmodel_locds(self):
		Ax,Ay, Bx, activ, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ay
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot2d]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.W [dot2d]',stack_start), (f'{_id}.B [dot2d]',stack_start + Ax*Bx)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot2d]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_dot2d = "Ax", "Ay", "Bx", "activ", "drop_rate"
	
	requiredposition_dot2d = 1,1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_dot2d(self, ystart, wstart, lstart, required):
		Ax, Ay, Bx, activ, drop_rate = required
		return Ax, Ay, Bx, activ, istart, ystart, wstart, lstart, drop_rate