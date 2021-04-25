#include "tensorflow/lite/micro/all_ops_resolver.h"
#include <stdarg.h>

extern void register_customs(tflite::AllOpsResolver *res);

// symbol needed inside this dll
int tflite::ErrorReporter::Report(const char *format, ...) {
  va_list va;
  va_start(va, format);
  vfprintf(stderr, format, va);
  va_end(va);
  return 0;
}

extern "C" TfLiteStatus register_custom(tflite::AllOpsResolver *res) {
  register_customs(res);
  return kTfLiteOk;
}
