class RMSPROP:
	name = "RMSPROP"

	description = '''v = beta * v + (1-beta) * grad(w)^2
	w -= alpha * grad(w) / sqrt(v)'''

	CONSTS = {
		'ALPHA' : 1e-5,
		'BETA' : 1e-4 
	}

	MIN_TEST_ECHOPES = 2

	def __init__(self, opti):
		self.train = opti.train

		self.hist = [0 for i in range(opti.train.mdl.weights * opti.train.sets)]

	def __del__(self):
		del self.hist

	def opti(self):
		mdl = self.train.mdl
		sets = self.train.sets
		ws = mdl.weights

		for s in range(sets):
			for w in range(ws):
				wpos = s*ws + w

				dw = self.train._meand[s*ws + w]				
				self.hist[wpos] = self.CONSTS['BETA']*self.hist[wpos] + (1 - self.CONSTS['BETA'])*dw**2

				self.train.w[wpos] -= self.CONSTS['ALPA'] * dw / sqrt(self.hist[wpos])