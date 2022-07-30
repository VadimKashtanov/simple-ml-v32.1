import struct as st

from .package.py.insts_arrays import INSTS
from inst import Inst

class Mdl:
	def __init__(self, insts:Inst, inputs, outputs, _vars, w, locds):
		#	Check if insts are all initialled

		for inst in insts:
			if 'params' in dir(inst):
				for k,v in inst.params.items():
					if v == None:
						raise Exception("Can't accept None param")
			else:
				raise Exception("Where is inst.`params` ? ")

		self.insts = insts

		self.inputs = inputs
		self.outputs = outputs

		self.vars = _vars
		self.weights = w

		self.locds = locds

		self.total = self.vars + self.inputs

	def check(self):
		for inst in self.insts:
			assert inst.check(params)

	def bin(self):
		bins = st.pack('N', len(_dict['insts']))

		for _id,params in self.insts:
			bins += st.pack('N', _id)
			bins += st.pack('N'*len(params), *params)

		bins += st.pack('NNNNN', self.inputs, self.outputs, self.vars, self.weights, self.locds)

		bins += st.pack('f'*len(self.w), *self.w)

		return bins