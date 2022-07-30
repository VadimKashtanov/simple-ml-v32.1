from math import exp

class SOFTMAX(Inst):
	_id = 3
	ID = 3

	name='SOFTMAX'
	params_names=['_len', 'input_start', 'ystart']

	def check(self):
		return all(i>=0 and int(i)==i for i in params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		_len, input_start, ystart = list(self.params.values())
		_sum = 0
		for i in range(_len):
			var[l*total + ystart + i] = exp(-var[l*total + input_start + i])
			_sum += var[l*total + ystart + i]
		for i in range(_len):
			var[l*total + ystart + i] /= _sum
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		_len, input_start, ystart = list(self.params.values())
		
		_sum = 0
		for i in range(_len):
			var[line*sets*total + _set*total + ystart + i] = exp(-var[line*sets*total + _set*total + input_start + i])
			_sum += var[line*sets*total + _set*total + ystart + i]
		for i in range(_len):
			var[line*sets*total + _set*total + ystart + i] /= _sum
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		_len, input_start, ystart = list(self.params.values())
	
		for i in range(_len):
			err = grad[line*sets*total + _set*total + ystart + i]
			for j in range(_len):
				grad[line*sets*total + _set*total + ystart + i] += err * var[line*sets*total + total*_set + ystart + i] * ((i == j) - var[line*sets*total + total*_set + ystart + j])
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		_len, input_start, ystart = list(self.params.values())
		return _len
	
	def buildstackmodel_weights(self):
		_len, input_start, ystart = list(self.params.values())
		return 0
	
	def buildstackmodel_locds(self):
		_len, input_start, ystart = list(self.params.values())
		return 0
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		_len, input_start, ystart = list(self.params.values())
		return [(f'{_id}.Y [softmax]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		_len, input_start, ystart = list(self.params.values())
		return []
	
	def labelstackmodel_locds(self,_id, stack_start):
		_len, input_start, ystart = list(self.params.values())
		return []
	
	### Setput Params Stack Model
	
	requiredforsetupparams_softmax = "_len",
	
	requiredposition_softmax = 1,0,0
	
	def setupparamsstackmodel_softmax(self, ystart, wstart, lstart, required):
		_len, = required
		return _len,istart,ystart