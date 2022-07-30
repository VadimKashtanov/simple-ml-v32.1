class ADADELTA:
	name = "ADADELTA"

	description = '''m = beta0*m + (1 - beta0)*grad(w)^2	
delta_w = - sqrt(v + 1e-8) / sqrt(m + 1e-8)
v = beta1*v + (1 - beta1)*delta_w^2

w -= delta_w
'''

	CONSTS = {
		'BETA0' : 1e-5,
		'BETA1' : 1e-5
	}

	MIN_TEST_ECHOPES = 2

	def __init__(self, opti):
		self.train = opti.train

		self.M = [0 for i in range(opti.train.mdl.weights * opti.train.sets)]
		self.V = [0 for i in range(opti.train.mdl.weights * opti.train.sets)]

	def __del__(self):
		del self.M, self.V

	def opti(self):
		mdl = self.train.mdl
		sets = self.train.sets
		ws = mdl.weights

		for s in range(sets):
			for w in range(ws):
				wpos = s*ws + w

				dw = self.train._meand[wpos]

				m = self.M[wpos] = self.CONSTS['BETA0']*self.M[wpos] + (1-self.CONSTS['BETA0'])*dw**2

				delta_w = -sqrt(self.V[wpos] + 1e-8) / sqrt(m + 1e-8)

				self.V[wpos] = self.CONSTS['BETA1']*self.V[wpos] + (1-self.CONSTS['BETA1'])*delta_w**2

				self.train.w[wpos] -= delta_w