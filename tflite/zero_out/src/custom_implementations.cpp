#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace zero_out {

extern TfLiteStatus Eval(TfLiteContext *, TfLiteNode *) { return kTfLiteOk; }

} // namespace zero_out

TfLiteRegistration *Register_ZeroOut(void) {
  static TfLiteRegistration res = {
      nullptr,
      nullptr,
      nullptr,
      zero_out::Eval,
  };
  return &res;
}

} // namespace micro
} // namespace ops
} // namespace tflite

void register_customs(tflite::AllOpsResolver *res) {
  res->AddCustom("ZeroOut", tflite::ops::micro::Register_ZeroOut());
}
