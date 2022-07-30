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

class DOT1DRECURENT(Inst):
	_id = 8
	ID = 8

	name='DOT1DRECURENT'

	#	Params : [Ax,At, Yx, activ, ist,yst,wst,lst, drate]
	#	At - de combien de lignes on va en arriere. Si At=1 =>  A=A[t-1]

	params_names=['Ax', 'At', 'Yx', 'activ', 'istart', 'ystart', 'wstart', 'lstart', 'drate']

	def check(self):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return activ < len(activate) and Ax > 0 and Yx > 0 and 100 >= drop_rate >= 0 and all(i>=0 and int(i)==i for i in params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
	
		for y in range(Yx):
			if At >= l:
				_sum = sum(var[(l-At)*total + input_start + i] * w[wstart + y*Ax + i] for i in range(Ax)) + w[wstart + Ax*Yx + y]
			else:
				_sum = w[wstart + Ax*Yx + y]
	
			var[l*total + ystart + y] = activate[activ](_sum)
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
	
		for y in range(Yx):
			if At >= line:
				_sum = sum(
					var[sets*total*(line-At) + _set*total + input_start + i] * w[ws*_set + wstart + Ax*y + i] for i in range(Ax)
					) + w[ws*_set + wstart + Ax*Yx + y]
			else:
				_sum = w[ws*_set + wstart + Ax*Yx + y]
	
			locd[sets*line*locds + _set*locds + locdstart + y] = localderiv[activ](_sum)
			var[sets*total*line + _set*total + ystart + y] = activate[activ](_sum)
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
	
		for y in range(Yx):
			dlds = locd[sets*line*locds + _set*locds + locdstart + y] * grad[sets*total*line + _set*total + ystart + y]
	
			#bias
			meand[ws*_set + wstart + Ax*Yx + y] += dlds
	
			if At >= line:
				for i in range(Ax):
					#var[sets*total*(line-At) + _set*total + input_start + i] * w[ws*_set + wstart + Ax*y + i]
					vpos = sets*total*(line-At) + _set*total + input_start + i
					wpos = ws*_set + wstart + Ax*y + i
	
					grad[vpos] += dlds * w[wpos]
					meand[wpos] += dlds * var[vpos]
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return Yx
	
	def buildstackmodel_weights(self):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return Ax*Yx + Yx
	
	def buildstackmodel_locds(self):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return Yx
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot1drecurent]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.W [dot1drecurent]',stack_start), (f'{_id}.B [dot1drecurent]',stack_start + Ax*Yx)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,At, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.Y [dot1drecurent]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_dot1drecurent = "Ax", "At", "Yx", "activ", "drop_rate"
	
	requiredposition_dot1drecurent = 1,1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_dot1drecurent(self, ystart, wstart, lstart, required):
		Ax, At, Yx, activ, drop_rate = required
		return Ax,At, Yx, activ, istart, ystart, wstart, lstart, drop_rate