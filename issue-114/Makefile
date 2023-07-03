CC = gcc
CPPFLAGS = -I$(ROCM_ROOT)/include -I$(ROCM_ROOT)/include/rocprofiler 
CFLAGS = -g -O2 -fPIC
LDFLAGS = -L$(ROCM_ROOT)/lib/rocprofiler -lrocprofiler64 

libissue-114.so: rocp_usr.o rocp_v1.o rocp_v2.o
	gcc -shared -o $@ $^

clean:
	rm -f *.o libissue-114.so
