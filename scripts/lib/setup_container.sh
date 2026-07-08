#!/usr/bin/env bash
# Shared in-container setup for TinkerCloud NeMo RL dev environments.
# Runs INSIDE the k8s pod or docker container after /tmp/code.tar.gz is in place.
# Single source of truth so the k8s and docker deployments cannot drift.
set -euo pipefail

cd /tmp && tar xzf code.tar.gz
mkdir -p /app /work /data/metadata /data/trajectories /data/.cache/huggingface

# TinkerCloud server -> /app (run via PYTHONPATH=/app python3 -m training)
rm -rf /app/training /app/tests /work/tinker_gmi /work/tinker-cookbook
mv /tmp/tinker-cloud/training /app/training
mv /tmp/tinker-cloud/tests /app/tests

# NeMo RL worker overlay onto the image's install (picked up at next create_model)
PKG=$(python3 -c 'import nemo_rl, os; print(os.path.dirname(nemo_rl.__file__))')
cp -r /tmp/RL/nemo_rl/. "$PKG/"

# SDK + cookbook editable installs
mv /tmp/tinker_gmi /work/tinker_gmi
mv /tmp/tinker-cookbook /work/tinker-cookbook
pip install -e /work/tinker_gmi --no-deps -q
# dev-mode bundles may lack .git -> setuptools-scm needs a pretend version
SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.0} \
  pip install -e /work/tinker-cookbook --no-deps -q

# deps missing from the nemo-rl image that --no-deps skips
# (distro: SDK; chz/termcolor/blobfile/tiktoken: cookbook;
#  sympy/pylatexenc/math-verify: math_rl recipe extras)
pip install -q distro chz termcolor blobfile tiktoken cloudpickle rich anyio \
  'httpx[http2]' sympy pylatexenc math-verify

python3 -c 'import tinker, tinker_cookbook, nemo_rl; print("imports OK")'
PYTHONPATH=/app python3 -c 'import training; print("training OK")'
echo SETUP_DONE
