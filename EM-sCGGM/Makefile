UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	# Mac OSX
	METISFILE=metis
	CXX=g++
	CXX=/usr/local/opt/llvm/bin/clang++
	CXXFLAGS=-fopenmp -fPIC -O3 -DNDEBUG

	# Advanced:
	# EM-sCGGM will be faster in Mac OSX if you install LAPACK using Homebrew:
	# brew install lapack
	# Then, add the following to LIBFLAGS:
	# -L/usr/local/opt/lapack/lib -framework Accelerate /usr/local/opt/lapack/lib/liblapacke.dylib
	# Then, define EIGEN_USE_LAPACKE and EIGEN_USE_BLAS 
	# in em_scggm.cc and m_step_yz.cc by uncommenting those lines in those files
else
	# GNU/Linux platform
	METISFILE=altmetis
	CXX=g++
	CXXFLAGS=-fopenmp -fPIC -O3 -DNDEBUG
endif

INCLUDEFLAGS=-I/usr/local/opt/llvm/include -I/usr/local/opt/llvm/include/c++/v1/ -I/usr/include -I../eigen337 -I/usr/local/include -I../metis-5.1.0/include
LIBFLAGS=-v -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -L/usr/local/lib -L../metis-5.1.0/lib

all: em_evaluate em_scggm

em_evaluate: em_evaluate.cc em_objective.o
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_objective.o em_evaluate.cc -o em_evaluate

em_scggm: em_scggm_run.cc em_scggm.o em_util.o m_step_xy.o m_step_yz.o smat.o sparse.o util.o em_objective.o
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} $^ -l${METISFILE} -o em_scggm

em_scggm.o: em_scggm.h em_scggm.cc em_util.o m_step_xy.o m_step_yz.o
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_scggm.cc -o em_scggm.o

em_objective.o: em_objective.h em_objective.cc
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} em_objective.cc -o em_objective.o

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

