#include "MNNPerfCounter.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

using namespace MNN::PerfCounter;

static void testA7xxMesaBindings() {
    CounterBinding binding;
    assert(resolveCounter(GpuVendor::Adreno, 740, "sp_cs_instructions", &binding));
    assert(binding.group == 10);
    assert(binding.selector == 56);
    assert(binding.slotsAvailable == 24);

    assert(resolveCounter(GpuVendor::Adreno, 740, "sp_gm_load_instructions", &binding));
    assert(binding.selector == 30);
    assert(resolveCounter(GpuVendor::Adreno, 740, "uche_vbif_read_beats_sp", &binding));
    assert(binding.group == 8);
    assert(binding.selector == 8);
}

static void testMaliProductFamiliesAndCounters() {
    DeviceInfo device;
    assert(identifyMali((static_cast<uint64_t>(0x0720) << 16), &device));
    assert(device.family == GpuFamily::Midgard);
    assert(std::strcmp(device.productName, "Mali-T720") == 0);

    assert(identifyMali((static_cast<uint64_t>(0xF) << 28) | (static_cast<uint64_t>(7) << 56) |
                           (static_cast<uint64_t>(3) << 32),
                       &device));
    assert(device.family == GpuFamily::Bifrost);
    assert(std::strcmp(device.productName, "Mali-G31") == 0);

    assert(identifyMali((static_cast<uint64_t>(0xF) << 28) | (static_cast<uint64_t>(9) << 56) |
                           (static_cast<uint64_t>(2) << 32),
                       &device));
    assert(device.family == GpuFamily::Valhall);
    assert(std::strcmp(device.productName, "Mali-G78") == 0);

    assert(identifyMali((static_cast<uint64_t>(0xF) << 28) | (static_cast<uint64_t>(12) << 56) |
                           (static_cast<uint64_t>(1) << 32),
                       &device));
    assert(device.family == GpuFamily::Arm5thGen);
    assert(std::strcmp(device.productName, "Mali-G620") == 0);

    CounterBinding binding;
    assert(resolveCounter(GpuVendor::Mali, device.productId, "gpu_active_cycles", &binding));
    assert(resolveCounter(GpuVendor::Mali, device.productId, "compute_tasks", &binding));
    assert(resolveCounter(GpuVendor::Mali, device.productId, "l2_any_lookup", &binding));
}

static void testUnknownProductIsRejected() {
    DeviceInfo device;
    assert(!identifyMali(0x12345678ULL, &device));
    CounterBinding binding;
    assert(!resolveCounter(GpuVendor::Adreno, 999, "sp_cs_instructions", &binding));
}

static void testUnavailableSessionIsControlled() {
    CounterSpec spec;
    spec.name = "gpu_active_cycles";
    const char* error = nullptr;
    DeviceInfo device;
    Session* session = Session::create(&spec, 1, &device, &error);
    if (session != nullptr) {
        assert(!session->start());
        assert(session->error() != nullptr);
        delete session;
    } else {
        assert(error != nullptr);
    }
}

int main() {
    testA7xxMesaBindings();
    testMaliProductFamiliesAndCounters();
    testUnknownProductIsRejected();
    testUnavailableSessionIsControlled();
    return 0;
}
