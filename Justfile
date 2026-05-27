set shell := ["bash", "-euo", "pipefail", "-c"]

build:
  @case "$(uname -s)" in \
    Linux|Darwin|MSYS*|MINGW*|CYGWIN*) ;; \
    *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;; \
  esac
  @for tool in cmake dxc; do \
    if ! command -v "$tool" >/dev/null 2>&1; then \
      echo "missing required tool: $tool" >&2; \
      exit 1; \
    fi; \
  done
  @if ! command -v spirv-val >/dev/null 2>&1; then \
    echo "optional tool missing: spirv-val; SPIR-V validation will be skipped" >&2; \
  fi
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --parallel
