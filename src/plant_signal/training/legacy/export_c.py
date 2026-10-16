"""Export the INT8 TFLite model as C/C++ source arrays."""

from __future__ import annotations

from plant_signal.config import C_MODEL_CC_PATH, C_MODEL_H_PATH, INT8_MODEL_PATH


BYTES_PER_LINE = 16
ARRAY_NAME = "g_plant_model_data"
ARRAY_LENGTH_NAME = "g_plant_model_data_len"


def _format_bytes(data: bytes) -> str:
    lines = []
    for offset in range(0, len(data), BYTES_PER_LINE):
        chunk = data[offset : offset + BYTES_PER_LINE]
        values = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"  {values},")
    return "\n".join(lines)


def main() -> None:
    if not INT8_MODEL_PATH.exists():
        raise FileNotFoundError(f"INT8 TFLite model not found: {INT8_MODEL_PATH}. Run convert_int8.py first.")

    data = INT8_MODEL_PATH.read_bytes()
    C_MODEL_CC_PATH.parent.mkdir(parents=True, exist_ok=True)

    header = f"""#ifndef PLANT_MODEL_DATA_H_
#define PLANT_MODEL_DATA_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char {ARRAY_NAME}[];
extern const unsigned int {ARRAY_LENGTH_NAME};

#ifdef __cplusplus
}}
#endif

#endif  // PLANT_MODEL_DATA_H_
"""

    source = f"""#include "plant_model_data.h"

const unsigned char {ARRAY_NAME}[] = {{
{_format_bytes(data)}
}};

const unsigned int {ARRAY_LENGTH_NAME} = {len(data)}u;
"""

    C_MODEL_H_PATH.write_text(header, encoding="utf-8", newline="\n")
    C_MODEL_CC_PATH.write_text(source, encoding="utf-8", newline="\n")

    print(f"Wrote: {C_MODEL_CC_PATH}")
    print(f"Wrote: {C_MODEL_H_PATH}")
    print(f"{ARRAY_LENGTH_NAME}: {len(data)} bytes")


if __name__ == "__main__":
    main()
