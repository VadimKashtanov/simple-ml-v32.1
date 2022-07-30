from math import log

ln = log

class CROSS_ENTROPY:
	name = "CROSS ENTROPY"

	description = '''Loss(g,w) = w*ln(g) + (1-w)*ln(1 - g)

dLoss/dg = w/g + (-1)*(1-w)/(1-g) =  w/g + (w - 1)/(1 - g)
		 = [w*(1 - g) + (w - 1)*g]/[(1 - g)*w] 
		 = [w - wg + wg - g]/[w - gw]
		 = [w - g]/[w - gw]

Score of a set = sum(output for lines for outputs) / (lines * outputs)
'''

	def __init__(self, opti):
		self.opti_class = opti
		self.train = opti.train

	def __del__(self):
		pass

	def loss(self):
		train = self.train
		mdl = train.mdl
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		scores = [0 for s in range(sets)]

		for s in range(sets):
			for l in range(lines):
				for o in range(outputs):
					pos = l*sets*total + s*total + outstart + o

					g = var[pos]
					w = out[l*outputs + o]
					scores[s] += w*ln(g) + (1-w)*ln(1 - g)

			scores[s] /= lines * outputs

		del self.opti_class.podium

		self.opti_class.podium = podium = [_set for score,_set in sorted(enumerate(scores), lambda x:x[1])]	#enumerate([0.1,0.2]) => [(0,0.1), (1,0.2)]

		for rank,(score,_set) in enumerate(podium):
			self.opti_class.set_score[_set] = score
			self.opti_class.set_rank[_set] = rank

	def dloss(self):
		train = self.train
		mdl = train.mdl
		data = train.data

		lines = data.lines
		sets = train.sets
		total = mdl.total
		outputs = data.outputs

		outstart = total - outputs

		for l in range(lines):
			for s in range(sets):
				for o in range(outputs):
					#get - want
					#so : get -= want, but for clarity we will leav it as it is
					get = train._var[(l*sets*total) + (s*_vars) + (outstart + o)]
					want = train.data.output[(l*outputs) + o]

					#= (w - g)/(w - gw)
					train._grad[(l*sets*_vars) + (s*total) + (outstart + o)] = (w - g)/(w - g*w)