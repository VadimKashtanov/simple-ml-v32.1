#
#	Gcc tools :
#		gdb, valgind
#

#device
	#DEBBUG = -G
#host
	DEBBUG = -g

#			-I$(0) est le package
ARGS = $(DEBBUG) -lm

all: optimize_mdl test

*.o:
	@ printf "[\033[35;1;41m***\033[0m] ================= KERNEL ===================\n"
	nvcc -c $(ARGS) $(shell find kernel -type f -name "*.cu") $(shell find package -type f -name "*.cu")

##
##	Les programmes sont les suivants
##
##		test_package			|	Tester toutes les instructions, les optimizeurs, les ...
##		optimize_mdl			|	Optimiser un model a partire d'un Data file. C'est tout.
##

test_package: *.o
	@ printf "[\033[35;1;41m***\033[0m] ============= PROGRAM : TEST PACKAGE ============\n"
	nvcc $(ARGS) *.o $(shell find programs/test_package -type f -name "*.cu") -o $@

optimize_mdl: *.o
	@ printf "[\033[35;1;41m***\033[0m] ================= PROGRAM : OPTIMIZE MODEL ===================\n"
	nvcc $(ARGS) *.o $(shell find programs/optimize_mdl -type f -name "*.cu") -o $@
