// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <cstdint>
struct QnnProbeEvent {uint32_t magic=0x514e4e50,type=0,unit=0,depth=0,parent=0;uint64_t value=0;};
struct QnnProbeMemory {uint32_t magic=0x4d454d50,input_shared=0,input_raw=0,output_shared=0,output_raw=0;uint64_t shared_bytes=0,raw_bytes=0;};
struct QnnProbeThread {uint32_t magic=0x43505550;uint64_t wall_ns=0,cpu_ns=0;};
struct QnnProbeCopy {uint32_t magic=0x434f5059,identical=0;uint64_t bytes=0;};
