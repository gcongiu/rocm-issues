#include "rocp.h"

int rocp_foo(int profiler_ver)
{
    if (profiler_ver == 1) {
        return rocpv1_foo();
    }
    return rocpv2_foo();
}
