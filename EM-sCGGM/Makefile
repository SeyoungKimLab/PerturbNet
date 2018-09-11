CXX=g++
CXXFLAGS=-fopenmp -O3 -fPIC -DNDEBUG 

######################## Mac OSX: #########################
# The native Apple clang compiler does not support OpenMP parallelization. 

# Thus, we recommend compiling with gcc.
# The easiest way to install gcc is to first install the Homebrew package manager:
# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# Then install gcc with Homebrew:
# brew install homebrew/versions/gcc49 --without-multilib
# Then uncomment the next two lines:
CXX=/usr/local/bin/g++-4.9
CXXFLAGS=-fopenmp -fPIC -O3 -DNDEBUG 

# Alternatively, compile and run without parallelization 
# by uncommenting the next two lines, without the openmp flag:
#CXX=g++
#CXXFLAGS=-fPIC -O3 -DNDEBUG

# EM-sCGGM will be faster in Mac OSX if you install LAPACK using Homebrew:
# brew install lapack
# Then, add the following to LIBFLAGS:
# -L/usr/local/opt/lapack/lib -framework Accelerate /usr/local/opt/lapack/lib/liblapacke.dylib
# Then, define EIGEN_USE_LAPACKE and EIGEN_USE_BLAS 
# in em_scggm.cc and m_step_yz.cc by uncommenting those lines in those files
###########################################################

INCLUDEFLAGS=-I/usr/include -I../eigen3 -I/usr/local/include -I../metis-5.1.0/include
LIBFLAGS=-L/usr/local/lib -L../metis-5.1.0/lib

all: em_evaluate em_scggm

em_evaluate: em_evaluate.cc
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_evaluate.cc -o em_evaluate

em_scggm: em_scggm_run.cc em_scggm.o em_util.o m_step_xy.o m_step_yz.o smat.o sparse.o util.o
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} $^ -lmetis -o em_scggm

em_scggm.o: em_scggm.h em_scggm.cc em_util.o m_step_xy.o m_step_yz.o
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_scggm.cc -o em_scggm.o

m_step_yz.o: m_step_yz.h m_step_yz.cc
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} m_step_yz.cc -o m_step_yz.o

m_step_xy.o: m_step_xy.h m_step_xy.cc sparse.o smat.o util.o
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} m_step_xy.cc -o m_step_xy.o

em_util.o: em_util.h em_util.cc
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_util.cc -o em_util.o

sparse.o: sparse.h sparse.cpp
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} sparse.cpp -o sparse.o

smat.o: smat.h smat.cpp
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} smat.cpp -o smat.o

util.o: util.h util.cc
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} util.cc -o util.o

clean:
	rm -rf *.o em_evaluate em_scggm *.mexa64 *.mexmaci64

