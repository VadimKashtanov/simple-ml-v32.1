class Inst:
	name = "Inst"

	params_names = []

	def __init__(self, preset=None):
		self.params = {}

		for i,param in enumerate(self.params_names):
			if preset != None:
				if type(preset[p]) is int and preset[p] >= 0:
					self.params[param] = preset[p]
				else:
					self.params[param] = None
			else:
				self.params[param] = None
		
		#p:(preset[p] if type(preset[p]) is int and preset[p] >= 0 else None) for i,p in enumerate(self.params_names)}

	def __getitem__(self, key):
		return params[key]

	def __setitem__(self, key, val):
		if not type(val) is int:
			raise Exception(f"Can't assign none positiv int to a parameter because kernel use `uint`. ({val} is not int type)")
		
		params[key] = val

	def check(self):
		for k,v in self.params:
			assert v != None