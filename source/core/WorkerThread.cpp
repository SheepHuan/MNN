#include "WorkerThread.hpp"
#include "KVMeta.hpp"
#include <thread>
#include <algorithm>
#include <mutex>
#include <vector>

#include <MNN/MNNDefine.h>
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace std;

namespace MNN {
namespace {

std::mutex& prefixDevicePrefetchMutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<PrefixDevicePrefetchSubmitter>& prefixDevicePrefetchSubmitters() {
    static std::vector<PrefixDevicePrefetchSubmitter> submitters;
    return submitters;
}

} // namespace

void registerPrefixDevicePrefetchSubmitter(PrefixDevicePrefetchSubmitter submitter) {
    if (submitter == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(prefixDevicePrefetchMutex());
    auto& submitters = prefixDevicePrefetchSubmitters();
    if (std::find(submitters.begin(), submitters.end(), submitter) == submitters.end()) {
        submitters.emplace_back(submitter);
    }
}

bool submitPrefixDevicePrefetch(KVMeta* meta) {
    if (meta == nullptr || !meta->prefix_device_prefetch || meta->prefix_request_id == 0) {
        return false;
    }
    std::vector<PrefixDevicePrefetchSubmitter> submitters;
    {
        std::lock_guard<std::mutex> lock(prefixDevicePrefetchMutex());
        submitters = prefixDevicePrefetchSubmitters();
    }
    bool submitted = false;
    for (auto submitter : submitters) {
        if (submitter != nullptr && submitter(meta)) {
            submitted = true;
        }
    }
    return submitted;
}

WorkerThread::WorkerThread(int numberThread) {
    for (int i=0; i<numberThread; ++i) {
        mWorkers.emplace_back([this]() {
            while (!mStop) {
                Task* f = nullptr;
                {
                    std::unique_lock<std::mutex> _l(mQueueMutex);
                    mCondition.wait(_l, [this] { return mStop || mTasks.size() > 0;});
                    if (mTasks.empty()) {
                        continue;
                    }
                    f = mTasks.front();
                    mTasks.pop();
                }
                f->content();
                delete f;
            }
        });
    }
}
WorkerThread::~WorkerThread() {
    {
        std::lock_guard<std::mutex> _l(mQueueMutex);
        mStop = true;
    }
    mCondition.notify_all();
    for (auto& worker : mWorkers) {
        worker.join();
    }
    // Complete Remain work
    while (!mTasks.empty()) {
        auto f = mTasks.front();
        f->content();
        mTasks.pop();
        delete f;
    }
}

bool WorkerThread::postTask(std::function<int()>&& task) {
    {
        AUTOTIME;
        std::unique_lock<std::mutex> _l(mQueueMutex);
        auto taskWrap = new Task;
        taskWrap->content = std::move(task);
        mTasks.push(taskWrap);
    }
    mCondition.notify_all();
    return true;
}

} // namespace MNN
