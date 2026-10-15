// SPDX-License-Identifier: MIT
// Copyright (c) 2026 LEPTONNW
#pragma once
// DFlash2-derived GDC equations: see NOTICE.md / Z Lab MIT attribution.
#include <mobile_dflash2/cpu_head.h>
#include <functional>
#include <unordered_map>
namespace mobile_dflash2 {
class CachedBody {
  struct Weight {std::vector<uint32_t> shape;std::vector<float> data;};
  std::unordered_map<std::string,Weight> weights_;
  std::vector<std::vector<float>> keys_,values_;
  size_t cached_=0;size_t feature_rows_=0;
  mutable CpuWorkers pool_;
  unsigned workers_=1;
  static constexpr size_t H=384,I=1536,D=64,KV=128,W=512,L=3;
  const Weight& weight(const std::string& name,std::initializer_list<uint32_t> shape)const {
    const auto& w=weights_.at(name);if(w.shape!=std::vector<uint32_t>(shape))throw std::runtime_error("CPU body shape mismatch: "+name);return w;
  }
  static float dot(const float* a,const float* b,size_t size) {
    float sum=0;size_t i=0;
#if defined(__ARM_NEON)
    float32x4_t x=vdupq_n_f32(0),y=x,z=x,t=x;
    for(;i+16<=size;i+=16){x=vfmaq_f32(x,vld1q_f32(a+i),vld1q_f32(b+i));y=vfmaq_f32(y,vld1q_f32(a+i+4),vld1q_f32(b+i+4));
      z=vfmaq_f32(z,vld1q_f32(a+i+8),vld1q_f32(b+i+8));t=vfmaq_f32(t,vld1q_f32(a+i+12),vld1q_f32(b+i+12));}
    sum=vaddvq_f32(vaddq_f32(vaddq_f32(x,y),vaddq_f32(z,t)));
#endif
    for(;i<size;++i)sum+=a[i]*b[i];return sum;
  }
  void linear(const std::string& name,const float* input,size_t rows,size_t in,size_t out,float* output)const {
    const auto& w=weight(name,{uint32_t(out),uint32_t(in)}).data;
    // Up to four query/new-context rows share each weight-vector read. This
    // avoids streaming a body matrix four times on a mobile memory bus.
    const unsigned count=out<128?1:workers_;
    auto work=[&](unsigned worker) {
    for(size_t o=out*worker/count;o<out*(worker+1)/count;++o) {
#if defined(__ARM_NEON)
      if(rows>1&&rows<=4) {
        float32x4_t sums[4];for(size_t r=0;r<rows;++r)sums[r]=vdupq_n_f32(0);
        size_t i=0;for(;i+4<=in;i+=4) {
          auto weights=vld1q_f32(w.data()+o*in+i);
          for(size_t r=0;r<rows;++r)sums[r]=vfmaq_f32(sums[r],vld1q_f32(input+r*in+i),weights);
        }
        for(size_t r=0;r<rows;++r) {
          float sum=vaddvq_f32(sums[r]);for(size_t j=i;j<in;++j)sum+=input[r*in+j]*w[o*in+j];output[r*out+o]=sum;
        }
        continue;
      }
#endif
      for(size_t r=0;r<rows;++r)output[r*out+o]=dot(input+r*in,w.data()+o*in,in);
    }
    };
    // Tiny projections cost more to dispatch than to compute. All other
    // outputs are disjoint; dot-product order within an output is unchanged.
    if(count==1)work(0);else pool_.run(count,work);
  }
  void norm(const std::string& name,const float* input,size_t rows,size_t size,float* output)const {
    const auto& w=weight(name,{uint32_t(size)}).data;
    for(size_t r=0;r<rows;++r){float sum=dot(input+r*size,input+r*size,size);float scale=1/std::sqrt(sum/size+1e-6f);
      for(size_t i=0;i<size;++i)output[r*size+i]=input[r*size+i]*scale*w[i];}
  }
  static void rotate(float* value,size_t heads,size_t position) {
    for(size_t h=0;h<heads;++h)for(size_t d=0;d<D/2;++d){float frequency=std::pow(10000.f,-float(2*d)/D);
      float angle=float(position)*frequency,cs=std::cos(angle),sn=std::sin(angle),a=value[h*D+d],b=value[h*D+d+D/2];
      value[h*D+d]=a*cs-b*sn;value[h*D+d+D/2]=b*cs+a*sn;}
  }
  void prepare_conv(const std::string& prefix,const std::vector<float>& hidden,size_t rows,std::vector<float>& prepared,std::vector<float>& kernels)const {
    kernels.resize(rows*96);linear(prefix+".kernel_projection.weight",hidden.data(),rows,H,96,kernels.data());
    apply_conv(prefix,hidden,rows,kernels,0,prepared);
  }
  void apply_conv(const std::string& prefix,const std::vector<float>& hidden,size_t rows,const std::vector<float>& kernels,size_t side,std::vector<float>& output)const {
    const auto& base=weight(prefix+".base_kernel",{2,2,H}).data;output.assign(rows*H,0);
    for(size_t r=0;r<rows;++r)for(size_t tap=0;tap<2&&tap<=r;++tap)for(size_t c=0;c<H;++c)
      output[r*H+c]+=hidden[(r-tap)*H+c]*(base[(side*2+tap)*H+c]+kernels[r*96+side*48+tap*24+c/16]);
  }
public:
  explicit CachedBody(const std::string& path) {
    std::ifstream f(path,std::ios::binary);char magic[8];uint32_t count=0;
    f.read(magic,8);f.read(reinterpret_cast<char*>(&count),4);
    if(!f||std::string(magic,8)!="DF2BODY1"||!count||count>200)throw std::runtime_error("Invalid CPU body header");
    for(uint32_t i=0;i<count;++i){uint32_t length=0,rank=0;uint64_t bytes=0;
      f.read(reinterpret_cast<char*>(&length),4);if(!length||length>200)throw std::runtime_error("Invalid weight name");
      std::string name(length,' ');f.read(name.data(),length);f.read(reinterpret_cast<char*>(&rank),4);
      if(rank>4)throw std::runtime_error("Invalid weight rank");Weight w;w.shape.resize(rank);
      f.read(reinterpret_cast<char*>(w.shape.data()),rank*4);f.read(reinterpret_cast<char*>(&bytes),8);
      size_t elements=1;for(auto dim:w.shape){if(!dim||dim>6144)throw std::runtime_error("Invalid weight dimension");elements*=dim;}
      if(bytes!=elements*4||bytes>40*1024*1024)throw std::runtime_error("Invalid weight byte count");w.data.resize(elements);
      f.read(reinterpret_cast<char*>(w.data.data()),bytes);if(!f||weights_.count(name))throw std::runtime_error("Truncated or duplicated body weight");
      for(float v:w.data)if(!std::isfinite(v))throw std::runtime_error("Nonfinite body weight");weights_.emplace(name,std::move(w));
    }
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("CPU body trailing bytes");
    keys_.resize(L,std::vector<float>(W*KV));values_=keys_;
  }
  size_t projected_feature_rows()const{return feature_rows_;}
  void set_workers(unsigned workers){if(!workers||workers>4)throw std::invalid_argument("Body workers must be 1..4");workers_=workers;}
  void reset(){cached_=0;feature_rows_=0;for(auto& k:keys_)std::fill(k.begin(),k.end(),0.f);for(auto& v:values_)std::fill(v.begin(),v.end(),0.f);}
  void update(size_t position,const std::function<const float*(size_t)>& feature) {
    if(position<cached_){cached_=0;feature_rows_=0;}
    cached_=std::max(cached_,position>W?position-W:0);
    std::vector<float> inputs(4*6144),projected(4*H),context(4*H),k(4*KV),v(4*KV),kn(4*KV);
    while(cached_<position){
      auto rows=std::min(size_t(4),position-cached_);
      for(size_t r=0;r<rows;++r)std::copy_n(feature(cached_+r),6144,inputs.data()+r*6144);
      linear("features.weight",inputs.data(),rows,6144,H,projected.data());norm("feature_norm.weight",projected.data(),rows,H,context.data());
      for(size_t layer=0;layer<L;++layer){auto prefix="layers."+std::to_string(layer)+".attention";
        linear(prefix+".k_proj.weight",context.data(),rows,H,KV,k.data());linear(prefix+".v_proj.weight",context.data(),rows,H,KV,v.data());
        norm(prefix+".k_norm.weight",k.data(),rows*2,D,kn.data());
        for(size_t r=0;r<rows;++r) {
          rotate(kn.data()+r*KV,2,cached_+r);
          std::copy_n(kn.data()+r*KV,KV,keys_[layer].data()+((cached_+r)%W)*KV);
          std::copy_n(v.data()+r*KV,KV,values_[layer].data()+((cached_+r)%W)*KV);
        }
      }cached_+=rows;feature_rows_+=rows;
    }
  }
  std::vector<float> forward(const float* embeddings,size_t position,size_t rows=4)const {
#if defined(RABI_DFLASH_EXPERIMENTAL_WIDTHS)
    if(position!=cached_||rows<4||rows>8)throw std::runtime_error("CPU cached context is not ready or width is unsupported");
#else
    if(position!=cached_||rows!=4)throw std::runtime_error("CPU cached context is not ready");
#endif
    std::vector<float> hidden(embeddings,embeddings+rows*H),normalized(rows*H),prepared,kernels,finished;
    for(size_t layer=0;layer<L;++layer){auto prefix="layers."+std::to_string(layer);auto attn=prefix+".attention";
      norm(prefix+".input_norm.weight",hidden.data(),rows,H,normalized.data());prepare_conv(prefix+".attention_conv",normalized,rows,prepared,kernels);
      std::vector<float> q(rows*H),qn(rows*H),k(rows*KV),kn(rows*KV),v(rows*KV),attention(rows*H,0),out(rows*H);
      linear(attn+".q_proj.weight",prepared.data(),rows,H,H,q.data());norm(attn+".q_norm.weight",q.data(),rows*6,D,qn.data());
      linear(attn+".k_proj.weight",prepared.data(),rows,H,KV,k.data());norm(attn+".k_norm.weight",k.data(),rows*2,D,kn.data());
      linear(attn+".v_proj.weight",prepared.data(),rows,H,KV,v.data());
      for(size_t r=0;r<rows;++r){rotate(qn.data()+r*H,6,position+r);rotate(kn.data()+r*KV,2,position+r);}
      size_t start=position>W?position-W:0;std::vector<float> scores(position-start+rows);
      for(size_t r=0;r<rows;++r)for(size_t h=0;h<6;++h){size_t kh=h/3,index=0;float maximum=-INFINITY;
        for(size_t absolute=start;absolute<position;++absolute){float score=absolute+W>position+r?dot(qn.data()+r*H+h*D,keys_[layer].data()+(absolute%W)*KV+kh*D,D)*.125f:-INFINITY;
          scores[index++]=score;maximum=std::max(maximum,score);}
        for(size_t b=0;b<rows;++b){float score=dot(qn.data()+r*H+h*D,kn.data()+b*KV+kh*D,D)*.125f;scores[index++]=score;maximum=std::max(maximum,score);}
        float denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
        index=0;auto* destination=attention.data()+r*H+h*D;
        for(size_t absolute=start;absolute<position;++absolute){float scale=scores[index++]/denominator;auto* source=values_[layer].data()+(absolute%W)*KV+kh*D;for(size_t d=0;d<D;++d)destination[d]+=scale*source[d];}
        for(size_t b=0;b<rows;++b){float scale=scores[index++]/denominator;auto* source=v.data()+b*KV+kh*D;for(size_t d=0;d<D;++d)destination[d]+=scale*source[d];}
      }
      linear(attn+".o_proj.weight",attention.data(),rows,H,H,out.data());apply_conv(prefix+".attention_conv",out,rows,kernels,1,finished);
      for(size_t i=0;i<hidden.size();++i)hidden[i]+=finished[i];
      norm(prefix+".post_norm.weight",hidden.data(),rows,H,normalized.data());prepare_conv(prefix+".mlp_conv",normalized,rows,prepared,kernels);
      std::vector<float> gate(rows*I),up(rows*I),down(rows*H);
      linear(prefix+".gate.weight",prepared.data(),rows,H,I,gate.data());linear(prefix+".up.weight",prepared.data(),rows,H,I,up.data());
      for(size_t i=0;i<gate.size();++i)gate[i]=gate[i]/(1+std::exp(-gate[i]))*up[i];
      linear(prefix+".down.weight",gate.data(),rows,I,H,down.data());apply_conv(prefix+".mlp_conv",down,rows,kernels,1,finished);
      for(size_t i=0;i<hidden.size();++i)hidden[i]+=finished[i];
    }
    norm("norm.weight",hidden.data(),rows,H,normalized.data());return normalized;
  }
};
}
