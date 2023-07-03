#include <stdio.h>
#include <rocprofiler.h>
#include "rocp.h"

int rocpv1_foo(void)
{
    fprintf(stdout, "rocprofiler version 1\n");
    return 0;
}
