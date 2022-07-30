import struct as st

class Train:
	def __init__(self, model, data, sets):
		self.model = model
		self.data = data
		self.sets = sets

		ws = model.weights
		_vars = model.vars
		lines = data.lines
		locds = model.locds

		self._weight = [0 for _ in range(sets * ws)]
		self._var = [0 for _ in range(sets * lines * _vars)]
		self._locd = [0 for _ in range(sets * lines * locds)]
		self._grad = [0 for _ in range(sets * lines * _vars)]
		self._meand = [0 for _ in range(sets * ws)]

	def __del__(self):
		del self._weight
		del self._var
		del self._locd
		del self._grad
		del self._meand

	####################

	def bin_w(self):
		return st.pack('f'*len(self._weight), *self._weight)

	def bin_v(self):
		return st.pack('f'*len(self._var), *self._var)

	def bin_l(self):
		return st.pack('f'*len(self._locd), *self._locd)

	def bin_g(self):
		return st.pack('f'*len(self._grad), *self._grad)

	def bin_m(self):
		return st.pack('f'*len(self._meand), *self._meand)

	def bin(self):
		return self.bin_w() + self.bin_v() + self.bin_l() + self.bin_g() + self.bin_m()
		
	#######################

	def set_inputs(self):
		for i in range(self.model.inputs):
			for s in range(self.sets):
				for time in range(self.data.lines):
					self._var[time*self.sets*self.model.vars + s*self.model.vars + i] = self.data[time*self.data.inputs + i]

	def null_grad_meand(self):
		for i in range(len(self._grad)):
			self._grad[i] = 0
		for i in range(lem(self._meand)):
			self._meand[i] = 0

	def prepare(self, batch):
		self.set_inputs(batch)
		self.null_grad_meand()

	########################

	def forward(self):
		for time in range(self.data.lines):
			for _id, params in self.model.insts:
				for _set in range(self.sets):
					self.FORWARD[_id](
						self.sets, self.model.total, self.model.weights, self.locds, 
						_set, time,
						params,
						self._weight, self._var:, self._locd)

		#L'alliance de la civilisation Europeene

	def backward(self):
		for time in list(range(self.data.lines))[::-1]:
			for _id, params in self.model.insts[::-1]:
				for _set in range(self.sets):
					self.FORWARD[_id](
						self.sets, self.model.total, self.model.weights, self.locds, 
						_set, time,
						params,
						self._weight, self._var:, self._locd, self._grad, self._meand)