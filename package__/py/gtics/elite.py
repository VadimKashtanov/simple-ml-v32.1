from .kernel.py.gtics import Gtic

def find_elites_int(elites, sets):
	if elites % sets == 0:
		return elites
	elites_sub = elites
	elites_sum = elites
	i = 0
	while 1:
		#	elites
		elites_sub -= 1
		elites_sum += 1

		#	Check if invalide
		if elites_sub == 0:
			assert 0
		if elites_sum == sets:
			assert 0

		#	Check 
		if elites_sub % sets == 0:
			return elites_sub
		if elites_sum % sets == 0:
			return elites_sum

def elite_mk(model, step,
		inp, out,
		w,
		var, locd,
		grad, meand,
		rank, scores,
		args):
	sets = model['sets']
	args_dico = {a.split('=')[0] : a.split('=')[1] for a in args}
	elites = args_dico['elites']

	if elites[-1] == '%':
		elites = int(elites[:-1])/100
	else:
		assert 0

	elites = find_elites_int(int(sets * elites), sets)

	return {
		'model' : model,
		'step'	: step,
		'inp'	: inp,
		'out'	: out,
		'w'		: w,
		'var'  	: var,
		'locd'	: locd,
		'grad'	: grad, 
		'meand'	: meand,
		'rank'	: rank,
		'scores': scores,

		'elites': elites
	}

def elite_free(_dico):
	pass

pseudo_randomf = lambda seed: ((123456*(seed+12345))%10000)/10000

def elite(_dico):
	n = int(_dico['model']['sets'] / _dico['model']['elites'])
	elites = _dico['elites']
	for elite in range(elites):
		for w in range(w):
			elite_w = _dico['w'][rank[elite]*_dico['weights'] + w]

			for _n in range(n):
				clone_w = rank[elites + elite*n + _n]*_dico['weights'] + w
				_dico['w'][clone_w] = elite_w + 0.01*(2*(pseudo_randomf(clone_w)-0.5))

class ELITE(Gtic):
	ARGS = {
		'elite_elites' : ,
		'elite_echopes' : ,
	}
	def select(self):
		opti = self.opti