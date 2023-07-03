ROCM_ROOT ?= /opt/rocm
ROCP_INC  ?= $(ROCM_ROOT)/include/rocprofiler
ROCP_LIB  ?= $(ROCM_ROOT)/lib
HSA_INC   ?= $(ROCM_ROOT)/include/hsa
HSA_LIB   ?= $(ROCM_ROOT)/lib
HIPCC     ?= $(ROCM_ROOT)/bin/hipcc

CXX      = $(HIPCC)
CPPFLAGS = -I$(ROCM_ROOT)/include -I$(HSA_INC) -I$(ROCP_INC)
CXXFLAGS = -g -O2 -fopenmp
LDFLAGS  = -L$(HSA_LIB) -lhsa-runtime64 -L$(ROCP_LIB) -lrocprofiler64 -ldl

issue-80: sampling.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f *.o issue-80
