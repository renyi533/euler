#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using namespace tensorflow::shape_inference;

REGISTER_OP("EulerHashFid")
    .Attr("use_locking: bool = false")
    .Input("ref: Ref(int64)")
    .Input("fids: int64")
    .Input("start: int64")
    .Input("end: int64")
    .Output("output_fids: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &x));
      c->set_output(0, c->input(1));
      return Status::OK();
    }); 
      
REGISTER_OP("HashToFid")
    .Attr("erase: bool = false")
    .Input("ref: Ref(int64)")
    .Input("fids: int64")
    .Input("start: int64")
    .Output("output_fids: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &x));
      c->set_output(0, c->input(1));
      return Status::OK();
    });       
} // namespace tensorflow
