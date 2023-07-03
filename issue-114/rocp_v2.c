#include <stdio.h>
#include <rocprofiler.h>
#include "rocp.h"

int rocpv2_foo(void)
{
    fprintf(stdout, "rocprofiler version 2\n");
    return 0;
}
