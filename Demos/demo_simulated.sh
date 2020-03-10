#!/bin/sh
../Mega-sCGGM/mega_scggm -y 3.2 -x 0.97 100 10000 100 20000 ../Examples/simulated_Y_n100_p20000_q10000.txt ../Examples/simulated_X_n100_p20000.txt simulatedLambda simulatedTheta simulatedStats
../Fast-sCGGM/fast_scggm -y 3.2 -x 0.97 100 10000 100 20000 ../Examples/simulated_Y_n100_p20000_q10000.txt ../Examples/simulated_X_n100_p20000.txt simLambda simTheta simStats
