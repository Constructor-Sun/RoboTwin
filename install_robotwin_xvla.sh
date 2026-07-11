#!/usr/bin/env bash
set -eo pipefail

ROBOTWIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/data1/liu/miniconda3/etc/profile.d/conda.sh"
XVLA_PREFIX="/data1/liu/miniconda3/envs/xvla"

if [ ! -f "$CONDA_SH" ]; then
  echo "Cannot find conda activation script: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate xvla

if [ "${CONDA_PREFIX:-}" != "$XVLA_PREFIX" ]; then
  echo "Refusing to install outside xvla. CONDA_PREFIX=${CONDA_PREFIX:-<unset>}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
PY="$XVLA_PREFIX/bin/python"

echo "Using Python: $("$PY" -c 'import sys; print(sys.executable)')"
echo "Installing RoboTwin requirements into xvla ..."
"$PY" -m pip install -r "$ROBOTWIN_DIR/script/requirements.txt"

echo "Installing CUDA 12 nvcc into xvla ..."
"$PY" -m pip install 'nvidia-cuda-nvcc-cu12==12.6.77' || "$PY" -m pip install nvidia-cuda-nvcc-cu12

SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
PIP_CUDA_NVCC_HOME="$SITE_PACKAGES/nvidia/cuda_nvcc"
CUDA_HOME="$XVLA_PREFIX"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "PyPI nvidia-cuda-nvcc-cu12 provides ptxas/NVVM but not nvcc; installing conda cuda-nvcc into xvla ..."
  conda install -p "$XVLA_PREFIX" -y -c nvidia cuda-nvcc=12.6 || \
    conda install -p "$XVLA_PREFIX" -y -c conda-forge cuda-nvcc=12.6
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "Cannot find xvla CUDA nvcc at: $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi

export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export C_INCLUDE_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/include:$PIP_CUDA_NVCC_HOME/include:$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/include:$PIP_CUDA_NVCC_HOME/include:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
NVIDIA_LIBS="$(find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -)"
export LIBRARY_PATH="$NVIDIA_LIBS:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

echo "Using nvcc: $(command -v nvcc)"
nvcc --version

echo "Patching sapien URDF loader ..."
SAPIEN_LOCATION="$("$PY" -m pip show sapien | awk '/^Location:/{print $2}')/sapien"
URDF_LOADER="$SAPIEN_LOCATION/wrapper/urdf_loader.py"
if [ -f "$URDF_LOADER" ]; then
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
else
  echo "Cannot find sapien URDF loader: $URDF_LOADER" >&2
  exit 1
fi

echo "Patching mplib planner ..."
MPLIB_LOCATION="$("$PY" -m pip show mplib | awk '/^Location:/{print $2}')/mplib"
PLANNER="$MPLIB_LOCATION/planner.py"
if [ -f "$PLANNER" ]; then
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
else
  echo "Cannot find mplib planner: $PLANNER" >&2
  exit 1
fi

echo "Installing curobo ..."
if [ ! -d "$ROBOTWIN_DIR/envs/curobo/.git" ]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$ROBOTWIN_DIR/envs/curobo"
fi
"$PY" -m pip install -e "$ROBOTWIN_DIR/envs/curobo" --no-build-isolation
"$PY" -m pip install warp-lang==1.12.0
"$PY" -m pip install setuptools==69.5.1

echo "Verifying imports ..."
"$PY" - <<'PY'
import importlib.util

mods = [
    "numpy",
    "pkg_resources",
    "sapien",
    "mplib",
    "torch",
    "warp",
    "curobo",
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing modules after install: {missing}")
print("RoboTwin xvla environment looks ready.")
PY
