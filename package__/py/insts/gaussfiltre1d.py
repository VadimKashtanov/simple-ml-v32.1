from math import exp

'''
y will be in [0;1]

		      [p0,p1,p2]	
[x0,x1,x2] -> [y0,y1,y2]

y0 = exp(-(x0+p0)^2)

dL/dx = (-2) * (x0+p0) * y0
dL/dp = (-2) * (x0+p0) * y0

locd = [-2*(x0+p0)*y0, -2*(x1+p1)*y1, -2*(x2+p2)*y2]

'''

'''	Var
len
'''
'''	Weights
len
'''
'''	Lods
len
'''

#Params : [len, istart,ystart,wstart,lstart]

class GAUSSFILTRE1D(Inst):
	_id = 6
	ID = 6

	name='GAUSSFILTRE1D'
	params_names=['_len', 'istart', 'ystart', 'wstart', 'lstart']

	def check(self):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return all(i >= 0 for i in params) and _len > 0
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
	
		_len, istart,ystart,wstart,lstart = list(self.params.values())
	
		inp = l*total + istart
		out = l*total + ystart
		p = wstart
	
		for i in range(_len):
			var[out + i] = exp(-(var[inp + i] + w[p + i])**2)
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		assert 0
		
		_len, istart,ystart,wstart,lstart = list(self.params.values())
	
		inp = line*sets*total + _set*total + istart
		out = line*sets*total + _set*total + ystart
		p = _set*wsize + wstart
	
		for i in range(_len):
			var[out + x] = exp(-(var[inp + i] + w[p + i])**2)
			locd[line*sets*locds + _set*locds + lstart + i] = -2 * (var[inp + i] + w[p + i]) * var[out + x]
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		assert 0
	
		_len, istart,ystart,wstart,lstart = list(self.params.values())
	
		inp = line*sets*total + _set*total + istart
		out = line*sets*total + _set*total + ystart
		ppos = _set*wsize + wstart
	
		for i in range(_len):
			'''dy = grad[out + i]
			y = var[out + i]
			x = var[inp + i]
			p = w[ppos + i]
	
			dw = dy * ((-2) * y * (x+p))'''
	
			dw = locd[line*sets*locds + _set*locds + lstart + i] *  grad[out + i]
	
			grad[inp + i] += dw
			meand[p + i] += dw
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return _len
	
	def buildstackmodel_weights(self):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return _len
	
	def buildstackmodel_locds(self):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return _len
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return [(f'{_id}.Y [gaussfiltre1d]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return [(f'{_id}.P [gaussfiltre1d]',stack_start)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		_len, istart,ystart,wstart,lstart = list(self.params.values())
		return [(f'{_id}.Y [gaussfiltre1d]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_gaussfiltre1d = "_len",
	
	requiredposition_gaussfiltre1d = 1,0,0,0,0 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_gaussfiltre1d(self, ystart, wstart, lstart, required):
		_len, = required
		return _len, istart,ystart,wstart,lstart