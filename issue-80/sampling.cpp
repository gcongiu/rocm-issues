/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <hsa/hsa.h>
#include <rocprofiler.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <omp.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int i = y * width + x;
    if (i < (width * height)) {
        a[i] = b[i] + c[i];
    }
}

using namespace std;

/* Data structures handled by init and shutdown */
#define MAX_DEVICE_COUNT (16)
hsa_agent_t agent_info_arr[MAX_DEVICE_COUNT];
unsigned int agent_info_arr_len;
rocprofiler_t *contexts[MAX_DEVICE_COUNT];
rocprofiler_properties_t properties = { NULL, 128, NULL, NULL };

static hsa_status_t _count_devices(hsa_agent_t agent, void *data)
{
    unsigned int *count = (unsigned int *) data;
    hsa_device_type_t type;
    hsa_status_t status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    assert(status == HSA_STATUS_SUCCESS);
    if (type == HSA_DEVICE_TYPE_GPU) {
        agent_info_arr[(*count)++] = agent;
    }
    return status;
}

static unsigned int _get_device_count(void)
{
    unsigned int count = 0;
    hsa_status_t status = hsa_iterate_agents(&_count_devices, &count);
    assert(status == HSA_STATUS_SUCCESS);
    return count;
}

static int init_sampling(void)
{
    rocprofiler_feature_t features[MAX_DEVICE_COUNT];
    unsigned int feature_count = 1;
    for (int i = 0; i < agent_info_arr_len; ++i) {
        features[i].kind = ROCPROFILER_FEATURE_KIND_METRIC;
        features[i].name = "SQ_WAVES";
    } 

    static int mode_set;
    for (int i = 0; i < agent_info_arr_len; ++i) {
        const uint32_t mode = (mode_set == 0) ?
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_CREATEQUEUE | ROCPROFILER_MODE_SINGLEGROUP :
            ROCPROFILER_MODE_STANDALONE | ROCPROFILER_MODE_SINGLEGROUP;
        mode_set = 1;
        hsa_status_t status = rocprofiler_open(agent_info_arr[i], features + i,
                                               feature_count, &contexts[i],
                                               mode, &properties);
        assert(status == HSA_STATUS_SUCCESS);
    }
    return 0;
}

static int start_sampling(void)
{
    for (int i = 0; i < agent_info_arr_len; ++i) {
        hsa_status_t status = rocprofiler_start(contexts[i], 0);
        assert(status == HSA_STATUS_SUCCESS);
    }
    return 0;
}

static int stop_sampling(void)
{
    for (int i = 0; i < agent_info_arr_len; ++i) {
        hsa_status_t status = rocprofiler_stop(contexts[i], 0);
        assert(status == HSA_STATUS_SUCCESS);
    }
    return 0;
}

static int shutdown_sampling(void)
{
    for (int i = 0; i < agent_info_arr_len; ++i) {
        hsa_status_t status = rocprofiler_close(contexts[i]);
        assert(status == HSA_STATUS_SUCCESS);
    }
    return 0;
}

int main(void)
{
    hsa_init();

    char *rocm_root = getenv("ROCM_ROOT");
    if (rocm_root == NULL) {
        fprintf(stderr, "No Rocm installation dir given\n");
        return -1;
    }

    char metrics_path[PATH_MAX];
    sprintf(metrics_path, "%s/%s", rocm_root, "lib/rocprofiler/metrics.xml");
    setenv("ROCP_METRICS", metrics_path, 1);

    int i;
    int errors;

    agent_info_arr_len = _get_device_count();
    omp_set_num_threads((int) agent_info_arr_len);
    
    init_sampling();
    start_sampling();

#pragma omp parallel
    { 
        hipSetDevice(omp_get_thread_num());

        hipDeviceProp_t devProp;
        hipGetDeviceProperties(&devProp, 0);
        cout << "Device " << omp_get_thread_num() << " System minor " << devProp.minor << endl;
        cout << "Device " << omp_get_thread_num() << " System major " << devProp.major << endl;
        cout << "Device " << omp_get_thread_num() << " agent prop name " << devProp.name << endl;
        cout << "Device " << omp_get_thread_num() << " hip Device prop succeeded " << endl ;

        float *deviceA, *hostA = (float*)malloc(NUM * sizeof(float));
        float *deviceB, *hostB = (float*)malloc(NUM * sizeof(float));
        float *deviceC, *hostC = (float*)malloc(NUM * sizeof(float));

        // initialize the input data
        for (i = 0; i < NUM; i++) {
            hostB[i] = (float)i;
            hostC[i] = (float)i*100.0f;
        }

        HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

        HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(vectoradd_float, 
                           dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                           dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                           0, 0,
                           deviceA, deviceB, deviceC, WIDTH, HEIGHT);

        hipDeviceSynchronize();
        HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        // verify the results
        errors = 0;
        for (i = 0; i < NUM; i++) {
            if (hostA[i] != (hostB[i] + hostC[i])) {
                errors++;
            }
        }
        if (errors!=0) {
            printf("FAILED: %d errors\n",errors);
        } else {
            printf ("PASSED!\n");
        }

        HIP_ASSERT(hipFree(deviceA));
        HIP_ASSERT(hipFree(deviceB));
        HIP_ASSERT(hipFree(deviceC));

        free(hostA);
        free(hostB);
        free(hostC);
    }

    stop_sampling();
    shutdown_sampling();
    hsa_shut_down();

    return errors;
}
