CXX = $(ROCM_ROOT)/bin/hipcc
CPPFLAGS = -I$(ROCM_ROOT)/include -I$(ROCM_ROOT)/include/hsa -I$(ROCM_ROOT)/include/rocprofiler 
CXXFLAGS = -g -O2 -fopenmp
LDFLAGS = -L$(ROCM_ROOT)/lib -lrocprofiler64 -lhsa-runtime64


issue-113: intercept.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f *.o issue-113
