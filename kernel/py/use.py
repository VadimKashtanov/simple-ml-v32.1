from model import Mdl

class Use:
	def __init__(self, mdl, lines):
		self.mdl = mdl
		
		self.lines = lines
		self._var = [0 for i in range(mdl.total * lines)]
	
	def set_inputs(self, inputs):
		for line in range(self.lines):
			for i in range(len(self.mdl.inputs)):
				_vars[self.total * line + i] = inputs[self.mdl.inputs + i]

	def forward(self, _vars, line):
		for inst in self.insts:
			inst.mdl(self.mdl.total, line, self._var, self.mdl.w)

	def bin(self):
		return st.pack('f'*len(self._var), *self._var)