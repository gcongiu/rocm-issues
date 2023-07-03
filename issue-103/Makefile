CC = gcc
CPPFLAGS = -I$(ROCM_ROOT)/include -I$(ROCM_ROOT)/include/hsa -I$(ROCM_ROOT)/include/rocprofiler 
CFLAGS = -g -O2
LDFLAGS = -L$(ROCM_ROOT)/lib -lrocprofiler64 


issue-103: iterate.o
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f *.o issue-103
