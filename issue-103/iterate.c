#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "hsa.h"
#include "rocprofiler.h"

static hsa_status_t
(*rocp_iterate_info_p)(const hsa_agent_t *,
                       rocprofiler_info_kind_t,
                       hsa_status_t (*)(const rocprofiler_info_data_t,
                                        void *),
                       void *);

typedef struct {
    hsa_agent_t agents[8];
    int count;
} hsa_agent_arr_t;

static hsa_status_t
get_agent_handle_cb(hsa_agent_t agent, void *agent_arr)
{
    hsa_device_type_t type;
    hsa_agent_arr_t *agent_arr_ = (hsa_agent_arr_t *) agent_arr;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        agent_arr_->agents[agent_arr_->count] = agent;
        ++agent_arr_->count;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t
count_ntv_events_cb(const rocprofiler_info_data_t info, void *count)
{
    fprintf(stdout, "calling %s\n", __func__);
    (*(int *) count) += info.metric.instances;
    return HSA_STATUS_SUCCESS;
}

int main(void)
{
    const char *rocplib = "librocprofiler64.so";
    char *rocp_path = getenv("ROCP_LIB_PATH");
    if (rocp_path == NULL) {
        fprintf(stderr, "ROCP_LIB_PATH undefined.\n");
        return EXIT_FAILURE;
    }

    char pathname[512] = { 0 };
    sprintf(pathname, "%s/%s", rocp_path, rocplib);

    void *rocp_lib = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (rocp_lib == NULL) {
        fprintf(stderr, "dlopen error in rocprofiler: %s.\n", dlerror());
        return EXIT_FAILURE;
    }

    rocp_iterate_info_p = dlsym(rocp_lib, "rocprofiler_iterate_info");     
    if (rocp_iterate_info_p == NULL) {
        fprintf(stderr, "dlsym error in rocprofiler: %s.\n", dlerror());
        return EXIT_FAILURE;
    }

    hsa_agent_arr_t agent_arr = { 0 };
    hsa_init();
    hsa_iterate_agents(get_agent_handle_cb, &agent_arr);

    int event_count = 0;
    for (int i = 0; i < agent_arr.count; ++i) {
        rocp_iterate_info_p(&agent_arr.agents[i],
                            ROCPROFILER_INFO_KIND_METRIC,
                            &count_ntv_events_cb,
                            &event_count);
    }

    fprintf(stdout, "counted events = %d\n", event_count);

    return 0;
}
