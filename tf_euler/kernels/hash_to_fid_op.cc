#include <atomic>
#include <cstdlib>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow
{
using namespace std;
class HashToFidOp : public OpKernel {
 public:
  explicit HashToFidOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("erase", &erase_));
    OP_REQUIRES(c, IsRefType(c->input_type(0)),
                errors::InvalidArgument("first input needs to be a ref type"));  
  }

  void Compute(OpKernelContext* c) override {
    DoCompute(c);
  }

 private:
  bool erase_;

  void DoCompute(OpKernelContext* c) {
    Tensor params = c->mutable_input(0, false);
    const int64 param_size = params.shape().dim_size(0);

    auto fids = c->input(1).flat<int64>();
    auto start = c->input(2).flat<int64>();

    auto start_ = start(0);

    OP_REQUIRES(c, params.IsInitialized(),
              errors::FailedPrecondition("Null ref for params"));
    OP_REQUIRES(c, TensorShapeUtils::IsVector(params.shape()),
              errors::InvalidArgument("params must be 1-D, got shape: ",
                                      params.shape().DebugString()));

    Tensor *out_fids_tensor = nullptr;
    OP_REQUIRES_OK(c,
                   c->allocate_output(0, {fids.dimension(0)}, &out_fids_tensor));
    auto out_fids = out_fids_tensor->flat<int64>();
    atomic<std::int64_t>* buckets = 
      reinterpret_cast<atomic<std::int64_t>*>(
          const_cast<char*>(params.tensor_data().data()));
          
    for (int64 i = 0; i < fids.dimension(0); ++i) {
      int64_t fid = fids(i);
      int64 s_idx = llabs(fid) % param_size;
      int64_t expected = 0;
      int64 try_cnt = 0;
      while (!atomic_compare_exchange_strong_explicit(
                                        buckets + s_idx, 
                                        &expected, 
                                        fid, 
                                        memory_order_seq_cst, 
                                        memory_order_seq_cst)) {
        if (expected == fid) {
          break;
        }
        expected = 0;
        s_idx = (s_idx + 1) % param_size;
        OP_REQUIRES(c, ++try_cnt < param_size/2,
              errors::FailedPrecondition("hash space low! extreme long loop found"));
      }

      if (erase_) {
        expected = fid;
        atomic_compare_exchange_strong(buckets + s_idx, &expected, (int64_t)0); 
      }
      out_fids(i) = s_idx + start_;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("HashToFid").Device(DEVICE_CPU),
            HashToFidOp);
} // namespace tensorflow

