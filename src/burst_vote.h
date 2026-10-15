// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <QnnInterface.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <dlfcn.h>
#include <stdexcept>
// A scoped public QNN performance request, released when this diagnostic ends.
class BurstVote {
  void* library_=nullptr;QnnHtpDevice_PerfInfrastructure_t infrastructure_{};
  uint32_t client_=0;bool active_=false;
public:
  void enable() {
    library_=dlopen("libQnnHtp.so",RTLD_NOW|RTLD_LOCAL);if(!library_)throw std::runtime_error("Cannot open QNN HTP performance interface");
    auto providers_fn=reinterpret_cast<decltype(&QnnInterface_getProviders)>(dlsym(library_,"QnnInterface_getProviders"));
    const QnnInterface_t** providers=nullptr;uint32_t count=0;
    if(!providers_fn||providers_fn(&providers,&count)!=QNN_SUCCESS)throw std::runtime_error("Cannot query QNN providers");
    const QnnInterface_t* provider=nullptr;
    for(uint32_t i=0;i<count;++i)if(providers[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR&&
      providers[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR){provider=providers[i];break;}
    if(!provider)throw std::runtime_error("QNN performance interface version differs");
    QnnDevice_Infrastructure_t opaque=nullptr;
    if(provider->QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&opaque)!=QNN_SUCCESS||!opaque)throw std::runtime_error("Cannot obtain HTP infrastructure");
    auto* htp=static_cast<QnnHtpDevice_Infrastructure_t*>(opaque);
    if(htp->infraType!=QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF)throw std::runtime_error("Unexpected HTP infrastructure");
    infrastructure_=htp->perfInfra;
    if(infrastructure_.createPowerConfigId(0,0,&client_)!=QNN_SUCCESS)throw std::runtime_error("Cannot create diagnostic power client");
    active_=true;
    QnnHtpPerfInfrastructure_PowerConfig_t config{};
    config.option=QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    auto& vote=config.dcvsV3Config;vote.contextId=client_;vote.setDcvsEnable=1;vote.dcvsEnable=0;
    vote.powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    vote.setSleepLatency=1;vote.sleepLatency=40;vote.setBusParams=1;vote.setCoreParams=1;
    vote.busVoltageCornerMin=vote.busVoltageCornerTarget=vote.busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    vote.coreVoltageCornerMin=vote.coreVoltageCornerTarget=vote.coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[]={&config,nullptr};
    if(infrastructure_.setPowerConfig(client_,configs)!=QNN_SUCCESS)throw std::runtime_error("Cannot set scoped burst vote");
  }
  ~BurstVote(){if(active_)infrastructure_.destroyPowerConfigId(client_);if(library_)dlclose(library_);}
};
