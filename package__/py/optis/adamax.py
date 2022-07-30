class ADAMAX:
	name = "ADAMAX"

	description = '''m = beta0*m + (1 - beta0)*grad(w)
u = max(beta1 * u, abs(grad(w)))
	
w -= alpha * m / (u * (1 - beta1^t))'''

	CONSTS = {
		'ALPHA' : 0.002,
		'BETA0' : 0.9,
		'BETA1' : 0.999
	}

	MIN_TEST_ECHOPES = 2

	def __init__(self, opti):
		self.train = opti.train

		self.M = [0 for i in range(opti.train.mdl.weights * opti.train.sets)]
		self.U = [0 for i in range(opti.train.mdl.weights * opti.train.sets)]

		self.echopes = 0

	def __del__(self):
		del self.M, self.U

	def opti(self):
		mdl = self.train.mdl
		sets = self.train.sets
		ws = mdl.weights

		for s in range(sets):
			for w in range(ws):
				wpos = s*ws + w

				dw = self.train._meand[wpos]

				m = self.M[wpos] = self.CONSTS['BETA0']*self.M[wpos] + (1-self.CONSTS['BETA0'])*dw
				u = self.U[wpos] = max(self.CONSTS['BETA1']*self.U[wpos], abs(dw))

				self.train.w[wpos] -= self.CONSTS['ALPHA'] * m / (u * (1 - self.CONSTS['BETA1']**self.echopes))

		#	Echopes need for update
		self.echopes += 1