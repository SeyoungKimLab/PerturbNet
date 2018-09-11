CXX=/usr/bin/g++
CXXFLAGS=-fPIC -O3 -DNDEBUG
INCLUDEFLAGS=-I/usr/include -I../eigen3 -I/usr/local/include
LIBFLAGS= 

all: fast_scggm fast_scggm.o

fast_scggm: fast_scggm_run.cc fast_scggm.o
	$(CXX) ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} $^ -o fast_scggm

fast_scggm.o: fast_scggm.h fast_scggm.cc
	${CXX} -c ${CXXFLAGS} ${INCLUDEFLAGS} ${LIBFLAGS} fast_scggm.cc -o fast_scggm.o

clean:
	rm -rf *.o fast_scggm *.mexa64 *.mexmaci64

