#include "driver_ioctl.h"

#include <stdio.h>
#include <stdint.h>

int main(void) {
    int fd = hpc_gpu_adreno_ioctl_open_gpu_device();
    if (fd < 0) {
        printf("open=%d\n", fd);
        return 1;
    }
    uint32_t gpu_id = hpc_gpu_adreno_ioctl_get_gpu_device_id(fd);
    printf("fd=%d gpu_id=%u\n", fd, gpu_id);
    if (gpu_id < 700 || gpu_id >= 800) {
        hpc_gpu_adreno_ioctl_close_gpu_device(fd);
        return 2;
    }
    const uint32_t groups[] = {10, 4, 8, 9, 1, 0};
    const uint32_t selectors[] = {56, 0, 0, 0, 0, 2};
    for (size_t i = 0; i < sizeof(groups) / sizeof(groups[0]); ++i) {
        int result = hpc_gpu_adreno_ioctl_activate_counter(fd, groups[i], selectors[i]);
        printf("activate group=%u selector=%u result=%d\n", groups[i], selectors[i], result);
    }
    hpc_gpu_adreno_ioctl_counter_read_counter_t counter;
    uint64_t value = 0;
    counter.group_id = 10;
    counter.countable_selector = 56;
    counter.value = 0;
    int read_result = hpc_gpu_adreno_ioctl_query_counters(fd, 1, &counter, &value);
    printf("read result=%d value=%llu\n", read_result, (unsigned long long)value);
    for (size_t i = 0; i < sizeof(groups) / sizeof(groups[0]); ++i) {
        printf("deactivate group=%u selector=%u result=%d\n", groups[i], selectors[i],
               hpc_gpu_adreno_ioctl_deactivate_counter(fd, groups[i], selectors[i]));
    }
    hpc_gpu_adreno_ioctl_close_gpu_device(fd);
    return 0;
}
