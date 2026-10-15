// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <condition_variable>
#include <atomic>
#include <exception>
#include <cerrno>
#include <functional>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#if defined(__ANDROID__)
#include <sched.h>
#include <unistd.h>
#endif
namespace mobile_dflash2 {
// One owner submits disjoint jobs; three workers remain asleep between rounds.
class CpuWorkers {
  inline static std::atomic<bool> fixed_affinity_{false};
  std::mutex submission_,mutex_;
  std::condition_variable begin_,done_;
  std::vector<std::thread> threads_;
  std::function<void(unsigned)> task_;
  std::exception_ptr error_;
  size_t epoch_=0,pending_=0;
  unsigned workers_=1;
  bool stopped_=false;
  static void pin(unsigned worker) {
#if defined(__ANDROID__)
    if(sysconf(_SC_NPROCESSORS_CONF)==8) {
      // Android may exclude the prime core from a foreground service's cpuset.
      // Restrict every worker to the performance cluster and let the scheduler
      // distribute it over the currently permitted cores; never use cores 0..3.
      cpu_set_t set;CPU_ZERO(&set);for(int core=4;core<8;++core)CPU_SET(core,&set);
      if(sched_setaffinity(0,sizeof(set),&set)!=0) {
        const int error=errno;cpu_set_t allowed;CPU_ZERO(&allowed);sched_getaffinity(0,sizeof(allowed),&allowed);
        std::string ids;for(int core=0;core<8;++core)if(CPU_ISSET(core,&allowed))ids+=std::to_string(core)+",";
        std::ifstream groups("/proc/thread-self/cgroup");std::string group((std::istreambuf_iterator<char>(groups)),{});
        throw std::runtime_error("CPU performance-cluster affinity failed, errno="+std::to_string(error)+", allowed="+ids+", cgroup="+group);
      }
      if(fixed_affinity_.load(std::memory_order_relaxed)) {
        cpu_set_t allowed;CPU_ZERO(&allowed);
        if(sched_getaffinity(0,sizeof(allowed),&allowed)!=0)throw std::runtime_error("Cannot inspect performance affinity");
        std::vector<int> cores;
        for(int core=7;core>=4;--core)if(CPU_ISSET(core,&allowed))cores.push_back(core);
        if(cores.empty())throw std::runtime_error("No permitted performance core");
        // Owner/RPC thread on the fastest permitted core, other workers on
        // distinct remaining cores. cpuset restrictions still apply.
        CPU_ZERO(&set);CPU_SET(cores[worker%cores.size()],&set);
        if(sched_setaffinity(0,sizeof(set),&set)!=0)throw std::runtime_error("Fixed performance affinity failed");
      }
    }
#else
    (void)worker;
#endif
  }
  void failed() {std::lock_guard<std::mutex> lock(mutex_);if(!error_)error_=std::current_exception();}
  void loop(unsigned id) {
    size_t seen=0;
    while(true) {
      std::unique_lock<std::mutex> lock(mutex_);
      begin_.wait(lock,[&]{return stopped_||epoch_!=seen;});
      if(stopped_)return;
      seen=epoch_;if(id>=workers_)continue;
      auto task=task_;lock.unlock();
      try {pin(id);task(id);}catch(...){failed();}
      lock.lock();if(--pending_==0)done_.notify_one();
    }
  }
public:
  static void fixed_affinity(bool enabled){fixed_affinity_.store(enabled,std::memory_order_relaxed);}
  static void pin_owner(){pin(0);}
  CpuWorkers() {
    try {for(unsigned id=1;id<4;++id)threads_.emplace_back([this,id]{loop(id);});}
    catch(...) {{std::lock_guard<std::mutex> lock(mutex_);stopped_=true;}begin_.notify_all();for(auto& t:threads_)t.join();throw;}
  }
  ~CpuWorkers() {{std::lock_guard<std::mutex> lock(mutex_);stopped_=true;}begin_.notify_all();for(auto& t:threads_)t.join();}
  CpuWorkers(const CpuWorkers&)=delete;
  void run(unsigned workers,const std::function<void(unsigned)>& task) {
    if(!workers||workers>4)throw std::invalid_argument("Use 1..4 performance CPU workers");
    std::lock_guard<std::mutex> submission(submission_);
    {std::lock_guard<std::mutex> lock(mutex_);task_=task;workers_=workers;pending_=workers-1;error_=nullptr;++epoch_;}
    begin_.notify_all();
    try {pin(0);task(0);}catch(...){failed();}
    std::unique_lock<std::mutex> lock(mutex_);done_.wait(lock,[&]{return pending_==0;});
    task_={};if(error_)std::rethrow_exception(error_);
  }
};
}
