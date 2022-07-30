from package.py.gtics_arrays import GTICS
from package.py.optis_arrays import OPTIS
from package.py.scores_arrays import SCORES

from mdl import Mdl
from data import Data
from train import Train
from opti import Opti_Class

class TEST_MDL:
	#
	#	Dans une version future je pourrait mettre le Score systeme et l'optimizer dans Train_t. Apres vient le probleme si je veut utiliser un meme optimizer sur un train.
	#	A voire si c'est une bonne idee ou pas, mais bon je pourrais extraire le train vers un autre train, mais ca fera une sorte de duplication pour rien.
	#	A voire
	#

	mdl = Mdl([])

	lines = 1
	sets = 1

	vsep = []
	wsep = []
	lsep = []

	def test(self):
		#	Init Model and a batch of Data
		mdl = self.mdl
		data = Data(
			1, self.lines, 
			[random() for _ in range(self.lines * mdl.inputs)], 
			[random() for _ in range(self.lines * mdl.outputs)])

		train = Train(mdl, data, self.sets)
		
		#	Binary information for tracking evoltion of the systeme (juste arrays to bin)
		bins = b''

		bins += mdl.bins() + data.bins()

		train.set_inputs()
		train.null_grad_meand()
		train.forward()

		for score in SCORES:
			opti_class = Opti_Class(train, score, OPTIS[0])	#any optis that will not be used but

			opti_class.dloss()
			opti_class.loss()

			bins += opti_class.bin_score()
			bins += train.bin_g()

			del opti_class

		for opti in OPTIS:
			opti_class = Opti_Class(train, SCORES[0], opti)	#any optis that will not be used but

			train.set_inputs()
			train.null_grad_meand()
			train.forward()
			opti_class.dloss()
			train.backward()
			opti_class.opti()

			bins += train.bin_w()

			del opti_class

		for gtic in GTICS:
			opti_class = Opti_Class(train, SCORES[0], opti[0])
			_gtic = gtic(train)

		return bins