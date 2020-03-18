#!/bin/sh
../Mega-sCGGM/mega_scggm -y 3.2 -x 0.97 100 1000 100 2000 ../Examples/simulated_Y_n100_p2000_q1000.txt ../Examples/simulated_X_n100_p2000.txt simulatedLambda simulatedTheta simulatedStats
../Fast-sCGGM/fast_scggm -y 3.2 -x 0.97 100 1000 100 2000 ../Examples/simulated_Y_n100_p2000_q1000.txt ../Examples/simulated_X_n100_p2000.txt simLambda simTheta simStats
