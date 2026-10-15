// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#if defined(__ANDROID__)
#include <vulkan/vulkan.h>
#include <array>
#include <cstring>
#include "draft_head_spv.h"
namespace mobile_dflash2 {
// One small dispatch per draft round. CPU prepares exact dynamic INT8 inputs;
// Vulkan computes the output projection; CPU keeps sequential token selection.
class GpuHead {
  VkInstance instance_=VK_NULL_HANDLE;
  VkPhysicalDevice physical_=VK_NULL_HANDLE;
  VkDevice device_=VK_NULL_HANDLE;VkQueue queue_=VK_NULL_HANDLE;
  VkDescriptorSetLayout layout_=VK_NULL_HANDLE;VkPipelineLayout pipeline_layout_=VK_NULL_HANDLE;
  VkPipeline pipeline_=VK_NULL_HANDLE;VkDescriptorPool descriptors_=VK_NULL_HANDLE;
  VkDescriptorSet set_=VK_NULL_HANDLE;VkCommandPool pool_=VK_NULL_HANDLE;
  VkCommandBuffer command_=VK_NULL_HANDLE;VkFence fence_=VK_NULL_HANDLE;
  struct Buffer {VkBuffer handle=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;size_t bytes=0;};
  std::array<Buffer,6> buffers_{};
  uint32_t hidden_=0,vocabulary_=0;
  std::string name_;
  static void check(VkResult result,const char* action){if(result!=VK_SUCCESS)throw std::runtime_error(std::string(action)+": "+std::to_string(result));}
  void cleanup()noexcept {
    if(device_){vkDeviceWaitIdle(device_);
      if(fence_)vkDestroyFence(device_,fence_,nullptr);
      if(pool_)vkDestroyCommandPool(device_,pool_,nullptr);
      if(pipeline_)vkDestroyPipeline(device_,pipeline_,nullptr);
      if(pipeline_layout_)vkDestroyPipelineLayout(device_,pipeline_layout_,nullptr);
      if(descriptors_)vkDestroyDescriptorPool(device_,descriptors_,nullptr);
      if(layout_)vkDestroyDescriptorSetLayout(device_,layout_,nullptr);
      for(auto& b:buffers_){if(b.mapped)vkUnmapMemory(device_,b.memory);if(b.handle)vkDestroyBuffer(device_,b.handle,nullptr);if(b.memory)vkFreeMemory(device_,b.memory,nullptr);}
      vkDestroyDevice(device_,nullptr);
    }
    if(instance_)vkDestroyInstance(instance_,nullptr);
  }
  void allocate(Buffer& b,size_t bytes){
    b.bytes=bytes;VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=bytes;info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(vkCreateBuffer(device_,&info,nullptr,&b.handle),"Create draft buffer");
    VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(device_,b.handle,&requirements);
    VkPhysicalDeviceMemoryProperties memory;vkGetPhysicalDeviceMemoryProperties(physical_,&memory);
    uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<memory.memoryTypeCount;++i)if((requirements.memoryTypeBits&(1u<<i))&&
      (memory.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==
       (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
    if(type==UINT32_MAX)throw std::runtime_error("No coherent GPU draft memory");
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};alloc.allocationSize=requirements.size;alloc.memoryTypeIndex=type;
    check(vkAllocateMemory(device_,&alloc,nullptr,&b.memory),"Allocate draft memory");
    check(vkBindBufferMemory(device_,b.handle,b.memory,0),"Bind draft memory");
    check(vkMapMemory(device_,b.memory,0,requirements.size,0,&b.mapped),"Map draft memory");
  }
public:
  GpuHead(uint32_t vocabulary,uint32_t hidden,const std::vector<int8_t>& weights,const std::vector<float>& scales):hidden_(hidden),vocabulary_(vocabulary) {
    try {
      if(hidden%4||weights.size()!=size_t(vocabulary)*hidden||scales.size()!=vocabulary)throw std::runtime_error("GPU head dimensions differ");
      VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="MobileDFlash2";app.apiVersion=VK_API_VERSION_1_1;
      VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};create.pApplicationInfo=&app;
      check(vkCreateInstance(&create,nullptr,&instance_),"Create draft Vulkan instance");
      uint32_t count=0;check(vkEnumeratePhysicalDevices(instance_,&count,nullptr),"Enumerate draft GPUs");
      std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance_,&count,devices.data()),"Read draft GPUs");
      uint32_t family=UINT32_MAX;
      for(auto gpu:devices){
        VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(gpu,&properties);
        if(properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU||properties.limits.maxStorageBufferRange<weights.size())continue;
        uint32_t n=0;vkGetPhysicalDeviceQueueFamilyProperties(gpu,&n,nullptr);std::vector<VkQueueFamilyProperties> families(n);vkGetPhysicalDeviceQueueFamilyProperties(gpu,&n,families.data());
        for(uint32_t i=0;i<n;++i)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT){physical_=gpu;family=i;name_=properties.deviceName;break;}
        if(physical_)break;
      }
      if(!physical_)throw std::runtime_error("No compatible Vulkan draft GPU");
      float priority=1;VkDeviceQueueCreateInfo queues{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};queues.queueFamilyIndex=family;queues.queueCount=1;queues.pQueuePriorities=&priority;
      VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};device.queueCreateInfoCount=1;device.pQueueCreateInfos=&queues;
      check(vkCreateDevice(physical_,&device,nullptr,&device_),"Create draft GPU device");vkGetDeviceQueue(device_,family,0,&queue_);
      const size_t bytes[]={weights.size(),scales.size()*4,size_t(3)*hidden,size_t(3)*4,size_t(vocabulary)*4,size_t(3)*vocabulary*4};
      for(size_t i=0;i<6;++i)allocate(buffers_[i],bytes[i]);
      std::memcpy(buffers_[0].mapped,weights.data(),weights.size());std::memcpy(buffers_[1].mapped,scales.data(),scales.size()*4);
      std::array<VkDescriptorSetLayoutBinding,6> bindings{};
      for(uint32_t i=0;i<6;++i){bindings[i].binding=i;bindings[i].descriptorCount=1;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
      VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=6;li.pBindings=bindings.data();check(vkCreateDescriptorSetLayout(device_,&li,nullptr,&layout_),"Create draft layout");
      VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,12};VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&layout_;pi.pushConstantRangeCount=1;pi.pPushConstantRanges=&push;
      check(vkCreatePipelineLayout(device_,&pi,nullptr,&pipeline_layout_),"Create draft pipeline layout");
      VkShaderModule shader=VK_NULL_HANDLE;VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=sizeof(rabi_draft_head_spv);si.pCode=rabi_draft_head_spv;
      check(vkCreateShaderModule(device_,&si,nullptr,&shader),"Create draft shader");
      VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.layout=pipeline_layout_;ci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=shader;ci.stage.pName="main";
      auto status=vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&ci,nullptr,&pipeline_);vkDestroyShaderModule(device_,shader,nullptr);check(status,"Compile draft GPU pipeline");
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6};VkDescriptorPoolCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};di.maxSets=1;di.poolSizeCount=1;di.pPoolSizes=&size;check(vkCreateDescriptorPool(device_,&di,nullptr,&descriptors_),"Create draft descriptor pool");
      VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=descriptors_;ai.descriptorSetCount=1;ai.pSetLayouts=&layout_;check(vkAllocateDescriptorSets(device_,&ai,&set_),"Allocate draft descriptors");
      std::array<VkDescriptorBufferInfo,6> infos{};std::array<VkWriteDescriptorSet,6> writes{};
      for(size_t i=0;i<6;++i){infos[i]={buffers_[i].handle,0,buffers_[i].bytes};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set_;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}
      vkUpdateDescriptorSets(device_,6,writes.data(),0,nullptr);
      VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool.queueFamilyIndex=family;pool.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;check(vkCreateCommandPool(device_,&pool,nullptr,&pool_),"Create draft command pool");
      VkCommandBufferAllocateInfo cmd{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cmd.commandPool=pool_;cmd.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cmd.commandBufferCount=1;check(vkAllocateCommandBuffers(device_,&cmd,&command_),"Allocate draft command buffer");
      VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(device_,&fence,nullptr,&fence_),"Create draft fence");
    }catch(...){cleanup();throw;}
  }
  ~GpuHead(){cleanup();}
  GpuHead(const GpuHead&)=delete;
  const std::string& device_name()const{return name_;}
  void project(const std::vector<int8_t>& queries,const std::vector<float>& scales,float* output,const std::vector<uint32_t>* active){
    if(queries.size()!=size_t(3)*hidden_||scales.size()!=3)throw std::runtime_error("GPU head requires three query rows");
    uint32_t count=active?uint32_t(active->size()):vocabulary_;
    if(!count||count>vocabulary_)throw std::runtime_error("Invalid GPU active vocabulary");
    std::memcpy(buffers_[2].mapped,queries.data(),queries.size());std::memcpy(buffers_[3].mapped,scales.data(),12);
    if(active)std::memcpy(buffers_[4].mapped,active->data(),count*4);
    check(vkResetCommandBuffer(command_,0),"Reset draft command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(command_,&begin),"Begin draft dispatch");
    VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};upload.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;upload.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&upload,0,nullptr,0,nullptr);
    vkCmdBindPipeline(command_,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_);vkCmdBindDescriptorSets(command_,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout_,0,1,&set_,0,nullptr);
    uint32_t parameters[]={count,hidden_/4,active?1u:0u};vkCmdPushConstants(command_,pipeline_layout_,VK_SHADER_STAGE_COMPUTE_BIT,0,12,parameters);
    const uint32_t columns=std::min(count,8192u);
    vkCmdDispatch(command_,columns,(count+columns-1)/columns,1);
    VkMemoryBarrier read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};read.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command_,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&read,0,nullptr,0,nullptr);
    check(vkEndCommandBuffer(command_),"End draft dispatch");check(vkResetFences(device_,1,&fence_),"Reset draft fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command_;check(vkQueueSubmit(queue_,1,&submit,fence_),"Submit draft head");
    check(vkWaitForFences(device_,1,&fence_,VK_TRUE,10000000000ULL),"Wait for draft head");
    std::memcpy(output,buffers_[5].mapped,size_t(count)*3*4);
  }
};
}
#endif
