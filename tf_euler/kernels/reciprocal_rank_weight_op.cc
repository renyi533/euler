/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"

namespace tensorflow {

class ReciprocalRankWeight: public OpKernel {
 public:
  explicit ReciprocalRankWeight(OpKernelConstruction* ctx): OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override;

 private:
  float GetWeight(int32 v);
  std::unordered_map<int32, float> weight_cache_;
};

float ReciprocalRankWeight::GetWeight(int32 v) {
  if (v <= 0) return 1;

  std::unordered_map<int32, float>::iterator iter;
  if ((iter = weight_cache_.find(v)) != weight_cache_.end()) {
    return iter->second;
  }
  
  float last_weight = GetWeight(v - 1);
  float weight = last_weight + 1.0f / float(v + 1);
  weight_cache_[v] = weight;
  return weight;
}

void ReciprocalRankWeight::Compute(OpKernelContext* ctx) {
  auto input = ctx->input(0);
  auto input_size = input.dim_size(0);

  auto input_data = input.flat<int32>();

  TensorShape output_shape;
  output_shape.AddDim(input_size);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto output_data = output->flat<float>();
  for (int i = 0; i < input_size; ++i) {
    float weight = GetWeight(input_data(i));
    output_data(i) = weight;
  }
}

REGISTER_KERNEL_BUILDER(Name("ReciprocalRankWeight").Device(DEVICE_CPU), ReciprocalRankWeight);

}  // namespace tensorflow
