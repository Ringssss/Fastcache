#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${ROOT}/sgl_kernel_shim.so}"

INCLUDES=$(python - <<'PY'
import torch
from torch.utils.cpp_extension import include_paths
print(" ".join("-I" + p for p in include_paths()))
PY
)
LIBS=$(python - <<'PY'
import torch
from torch.utils.cpp_extension import library_paths
print(" ".join("-L" + p for p in library_paths()))
PY
)
RPATHS=$(python - <<'PY'
import torch
from torch.utils.cpp_extension import library_paths
print(":".join(library_paths()))
PY
)

g++ -shared -fPIC -O2 "${ROOT}/sgl_kernel_shim.cc" -o "${OUT}" \
  ${INCLUDES} ${LIBS} -lc10 -ltorch -ltorch_cpu -Wl,-rpath,${RPATHS}

echo "Built: ${OUT}"
