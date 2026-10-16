/*
 * MCU-side preprocessing example for CH32H417 + TensorFlow Lite Micro.
 *
 * Fill the constants below after running train.py and convert_int8.py:
 * - PLANT_PREPROCESS_SCALE comes from artifacts/preprocess.json ("scale").
 * - TFLITE_INPUT_SCALE and TFLITE_INPUT_ZERO_POINT come from convert_int8.py.
 *
 * This file uses no dynamic memory and keeps the operation equivalent to
 * dataset.py: subtract per-window mean, divide by the fixed training scale,
 * then quantize to int8 using the TFLite input quantization parameters.
 */

#include <stdint.h>

#define PLANT_WINDOW_SIZE 128

/* TODO: replace with artifacts/preprocess.json scale after training. */
#define PLANT_PREPROCESS_SCALE (1.0f)

/* TODO: replace with convert_int8.py input scale and zero_point after conversion. */
#define TFLITE_INPUT_SCALE (1.0f)
#define TFLITE_INPUT_ZERO_POINT (0)

static int8_t clamp_int8(int32_t value) {
  if (value > 127) {
    return 127;
  }
  if (value < -128) {
    return -128;
  }
  return (int8_t)value;
}

static int32_t round_float_to_int32(float value) {
  if (value >= 0.0f) {
    return (int32_t)(value + 0.5f);
  }
  return (int32_t)(value - 0.5f);
}

int plant_preprocess_adc_window(const int32_t adc_samples[PLANT_WINDOW_SIZE],
                                int8_t model_input[PLANT_WINDOW_SIZE]) {
  int i;
  int64_t sum = 0;
  float mean;

  if (adc_samples == 0 || model_input == 0) {
    return -1;
  }
  if (PLANT_PREPROCESS_SCALE <= 0.0f || TFLITE_INPUT_SCALE <= 0.0f) {
    return -2;
  }

  for (i = 0; i < PLANT_WINDOW_SIZE; ++i) {
    sum += adc_samples[i];
  }
  mean = (float)sum / (float)PLANT_WINDOW_SIZE;

  for (i = 0; i < PLANT_WINDOW_SIZE; ++i) {
    float centered = (float)adc_samples[i] - mean;
    float normalized = centered / PLANT_PREPROCESS_SCALE;
    int32_t quantized =
        round_float_to_int32(normalized / TFLITE_INPUT_SCALE) +
        (int32_t)TFLITE_INPUT_ZERO_POINT;
    model_input[i] = clamp_int8(quantized);
  }

  return 0;
}
