UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	# Mac OSX
	METISFILE=metis
	# The native Apple clang compiler does not support OpenMP parallelization. 
	# Thus, we recommend compiling with gcc.
	# The easiest way to install gcc is to first install the Homebrew package manager:
	# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	# Then install gcc with Homebrew:
	# brew install homebrew/versions/gcc49 --without-multilib
	# Then uncomment the next two lines:
	# CXX=/usr/local/bin/g++-4.9
	# CXXFLAGS=-fopenmp -fPIC -O3 -DNDEBUG

	# Alternatively, compile and run without parallelization 
	# by uncommenting the next two lines, without the openmp flag:
	CXX=g++
	CXXFLAGS=-fPIC -O3 -DNDEBUG
else
	# GNU/Linux platform
	METISFILE=altmetis
	CXX=g++
	CXXFLAGS=-fopenmp -fPIC -O3 -DNDEBUG
endif

INCLUDEFLAGS=-I/usr/include -I/usr/local/include -I../metis-5.1.0/include
LIBFLAGS=-L/usr/local/lib -L../metis-5.1.0/lib -lm

all: mega_scggm mega_scggm.o sparse.o smat.o util.o

mega_scggm: mega_scggm_run.cpp mega_scggm.o smat.o sparse.o util.o
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} $^ -l${METISFILE} -o mega_scggm

mega_scggm.o: mega_scggm.h mega_scggm.cpp sparse.h smat.h util.h
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} mega_scggm.cpp -o mega_scggm.o

sparse.o: sparse.h sparse.cpp
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} sparse.cpp -o sparse.o

smat.o: smat.h smat.cpp
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} smat.cpp -o smat.o

util.o: util.h util.cpp
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} util.cpp -o util.o

clean:
	rm -rf *.pyc *.o mega_scggm *.mexa64 *.mexmaci64

