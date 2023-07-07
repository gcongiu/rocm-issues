#include <stdio.h>
#include <v2/rocprofiler.h>
#include "rocp.h"

int rocpv2_foo(void)
{
    fprintf(stdout, "rocprofiler version 2\n");
    return 0;
}
