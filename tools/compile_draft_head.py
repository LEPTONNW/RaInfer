"""Validate the Vulkan 1.1 integer-only draft head and embed SPIR-V."""
import subprocess,sys,struct
from pathlib import Path
glslc,validator,source,output=map(Path,sys.argv[1:])
output.parent.mkdir(parents=True,exist_ok=True)
binary=output.with_suffix('.spv')
subprocess.run([str(glslc),'-O','--target-env=vulkan1.1','--target-spv=spv1.3',str(source),'-o',str(binary)],check=True)
subprocess.run([str(validator),'--target-env','vulkan1.1',str(binary)],check=True)
data=binary.read_bytes();words=struct.unpack('<'+'I'*(len(data)//4),data)
output.write_text('#pragma once\n#include <cstdint>\ninline const uint32_t rabi_draft_head_spv[]={'+','.join(map(str,words))+'};\n',encoding='utf-8')
