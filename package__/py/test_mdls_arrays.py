TEST_MODELS = [								#	Explain what instructions does
	"TEST_MODEL_DOT1D",					# Dot product of a vector with a Bx*Ax weight
	"TEST_MODEL_DOT2D",					# Dot product of Ax*Ay and Bx*Ax
	"TEST_MODEL_KCONVL33SAMEPOOL22MAX",	# 3x3 Kernel convolution with 2x2 max pooling
	"TEST_MODEL_SOFTMAX",				# e^(-x)/sum(e^(-x))
	"TEST_MODEL_LSTM1D",				# vector lsmt
	"TEST_MODEL_LSTM2D",				# matrix lsmt
	"TEST_MODEL_GAUSSFILTRE1D",			# gauss filtre with 1 param for 1 pixel of vector
	"TEST_MODEL_GAUSSFILTRE2D",			# gauss filtre with 1 param for Ay pixels of matrix
	"TEST_MODEL_DOT1DRECURENT",			# Dot1d but with a past line/time input 	(ex : [t-1])
	"TEST_MODEL_DOT2DRECURENT",			# Dot2d but with a past line/time input   (ex : [t-1])
	"TEST_MODEL_DOTGAUSSFILTRE2D",		# Dot2d but instead of sum of `x*w`, we have sum of `exp(-(x^2) + w)`

	#je peux ajouter des model plus complexe, des cnn completes, des models pour xor, ...
]

test_models = []

for test_mdl in TEST_MODELS:
	exec(f"from .py.test_mdls.{test_mdl.lower()} import {test_mdl}")
	exec(f"test_models += [{test_mdl}]")