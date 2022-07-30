import struct as st

class Data:
	def __init__(self, batchs, lines, _input, output):
		self.batchs = batchs
		self.lines = lines
		self.inputs = int(len(_input)/(lines*batchs))
		self.outputs = int(len(output)/(lines*batchs))

		self.input = _input
		self.output = output

	def bin(self):
		bins = st.pack('NNNN', self.batchs, self.lines, self.inputs, self.outputs)

		bins += st.pack('f'*(self.batchs * self.lines * self.inputs), *self.input)
		bins += st.pack('f'*(self.batchs * self.lines * self.outputs), *self.output)

		return bins