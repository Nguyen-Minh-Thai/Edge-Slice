#!/usr/bin/env bash
set -euo pipefail

# Build a local, no-sudo Python 3.14 Torch/Numpy runtime from Ubuntu .deb
# packages.  This is useful in restricted Ubuntu containers where sudo apt
# install cannot authenticate but apt-get download is allowed.

DEB_DIR="${EDGESLIDE_DEB_DIR:-/tmp/edgeslide-debs}"
APT_ROOT="${EDGESLIDE_APT_ROOT:-/tmp/edgeslide-apt-root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PACKAGES=(
  python3-numpy
  python3-pytest
  python3-torch
  libblas3
  liblapack3
  libtorch2.9
  libopenmpi40
  libprotobuf32t64
  libtensorpipe0
  libevent-core-2.1-7t64
  libevent-pthreads-2.1-7t64
  libfabric1
  libhwloc15
  libnl-3-200
  libpmix2t64
  libucc1
  libucx0
  libhwloc-plugins
  libgfortran5
  librdmacm1t64
  libmunge2
  libxnnpack0.20241108
  libcpp-httplib0.26
  libcpuinfo0
  libdnnl3.6
  libfmt10
  libgloo0
  libonnx1t64
  libpthreadpool0
  libsleef3
  libhiredis1.1.0
  python3-typing-extensions
  python3-filelock
  python3-fsspec
  python3-jinja2
  python3-networkx
  python3-packaging
  python3-psutil
  python3-requests
  python3-setuptools
  python3-six
  python3-sympy
  python3-yaml
  python3-iniconfig
  python3-pluggy
  python3-pygments
)

mkdir -p "$DEB_DIR" "$APT_ROOT"
cd "$DEB_DIR"
apt-get download "${PACKAGES[@]}"

for deb in "$DEB_DIR"/*.deb; do
  dpkg-deb -x "$deb" "$APT_ROOT"
done

PYROOT="$APT_ROOT/usr/lib/python3/dist-packages"
LIBROOT="$APT_ROOT/usr/lib/aarch64-linux-gnu"
TORCH_LIB="$PYROOT/torch/lib"
export PYTHONPATH="$PYROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$TORCH_LIB:$LIBROOT:$LIBROOT/blas:$LIBROOT/lapack:$APT_ROOT/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cat <<EOF
Local Ubuntu Torch environment ready.

Run this before tests:
  export PYTHONPATH="$PYROOT\${PYTHONPATH:+:\$PYTHONPATH}"
  export LD_LIBRARY_PATH="$TORCH_LIB:$LIBROOT:$LIBROOT/blas:$LIBROOT/lapack:$APT_ROOT/usr/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

EOF

"$PYTHON_BIN" - <<'PY'
import numpy
import torch
print('numpy', numpy.__version__)
print('torch', torch.__version__)
print('torch_sum', float(torch.randn(2, 3).sum()))
PY

