#!/usr/bin/env bash
# One-key deploy of a TinkerCloud dev environment (k8s pod or docker container).
#
#   deploy_tinkercloud.sh [--profile nemo_rl|bionemo|megatron_bridge|miles] \
#                         [--source dev|git] [--target k8s|docker] [--code-only]
#
# Profiles (cased by BASE IMAGE — the images ship incompatible stacks)
#   nemo_rl  (default) NeMo RL server backend. Image nvcr.io/nvidia/nemo-rl:*.
#            Overlays RL/nemo_rl, starts a Ray head + `python3 -m training`
#            (TINKERCLOUD_BACKEND=nemo_rl), health-checks :8000. GPUS=4, master-02.
#   bionemo  cu12 NeMo2 exploration env (feature 004 P5). Image
#            nvcr.io/nvidia/clara/bionemo-framework:* — hyena Megatron-LM + TE +
#            bionemo.evo2 + NeMo2. Env only (code + scripts/evo2 helpers, no server).
#   megatron_bridge  cu13 recipe env (feature 004 P5, the FAITHFUL Evo2 classifier).
#            Image nvcr.io/nvidia/pytorch:26.04-py3. Builds the bionemo-recipes
#            evo2_megatron recipe (megatron-bridge v0.4.1) so the megatron_bridge
#            backend can import megatron.bridge + evo2_classifier. Env only until
#            that backend lands (stubs today). GPUS=2, master-03.
#   miles    Miles/Slime backend. Image is the private GMI GAR build of
#            github.com/GavinZhu-GMI/miles (miles + Megatron/SGLang/TE pre-installed);
#            overlays only the server code, starts Ray head + `python3 -m training`
#            (TINKERCLOUD_BACKEND=miles), health-checks :8000. GPUS=4, master-02.
#            Needs a pull secret for the private image: IMAGE_PULL_SECRET (default
#            gcp-secret) must exist in namespace $NS.
#   Per-profile defaults (IMAGE, GPUS, NODE, BACKEND) are overridable by env vars.
#   See specs/004-bionemo-classification/P5-TINKER-BACKEND.md.
#
# Sources
#   dev  (default when run inside the tinker-nemorl monorepo)
#        Bundle the LOCAL working trees — including uncommitted changes — of
#        tinker-cloud, RL, tinker_gmi, tinker-cookbook (monorepo siblings).
#   git  Self-contained: shallow-clone all four repos at pinned refs. Needs no
#        monorepo checkout; this is the user-facing path.
#        Override refs/URLs via env: TINKER_CLOUD_REPO/TINKER_CLOUD_REF,
#        RL_REPO/RL_REF, TINKER_GMI_REPO/TINKER_GMI_REF, COOKBOOK_REPO/COOKBOOK_REF.
#
# Targets (mirrored: same image, same in-container setup, only transport differs)
#   k8s     Pod on the kubecluster. Env: KUBECONFIG, NS, POD, NODE, GPUS.
#   docker  Local container on a GPU box. Env: CONTAINER, DATA_DIR, GPUS,
#           DOCKER (e.g. DOCKER="sg docker -c" when the shell lacks docker group).
#
# --code-only  Skip pod/container creation; redeploy code + restart server.
#
# HF token: HF_TOKEN_FILE=<path> (copied to /tmp/hf_token.txt in the target;
# if unset, an existing /tmp/hf_token.txt inside the target is reused).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${PROFILE:-nemo_rl}"
SOURCE=dev
TARGET=k8s
CODE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --code-only) CODE_ONLY=1; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# --- profile: case by base image (the two images' stacks are incompatible) --
# Sets per-profile DEFAULTS; every one is overridable by the same-named env var.
case "$PROFILE" in
  nemo_rl)
    IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.5.0}"
    BACKEND="${BACKEND:-nemo_rl}"
    GPUS="${GPUS:-4}"
    NODE="${NODE:-master-02}"
    POD="${POD:-tinkercloud-nemorl-m02}"
    CONTAINER="${CONTAINER:-tinkercloud-nemorl}"
    RUN_SERVER=1 ;;   # Ray head + `python3 -m training` + health check
  bionemo)
    IMAGE="${IMAGE:-nvcr.io/nvidia/clara/bionemo-framework:2.7.1}"
    BACKEND="${BACKEND:-automodel}"
    GPUS="${GPUS:-2}"
    NODE="${NODE:-master-03}"
    POD="${POD:-bionemo-evo2-p5}"
    CONTAINER="${CONTAINER:-bionemo-evo2-p5}"
    RUN_SERVER=0 ;;   # env only (cu12 NeMo2 exploration env): code + evo2 helpers
  megatron_bridge)
    # cu13 recipe env: builds bionemo-recipes evo2_megatron (megatron-bridge) so
    # the megatron_bridge backend can import megatron.bridge + the Evo2 classifier.
    IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
    BACKEND="${BACKEND:-megatron_bridge}"
    GPUS="${GPUS:-2}"
    NODE="${NODE:-master-03}"
    POD="${POD:-evo2-recipe}"
    CONTAINER="${CONTAINER:-evo2-recipe}"
    RUN_SERVER=0 ;;   # env only until the megatron_bridge backend lands (stubs today)
  miles)
    # Miles/Slime backend. Miles + its Megatron/SGLang/TE stack are pre-installed
    # in the GMI base image (built from github.com/GavinZhu-GMI/miles); we overlay
    # only the tinker-cloud server code. The image is a PRIVATE GAR repo, so the
    # pod needs an imagePullSecret (IMAGE_PULL_SECRET, default gcp-secret).
    IMAGE="${IMAGE:-us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/miles_dev-202511120a-dev:latest}"
    BACKEND="${BACKEND:-miles}"
    GPUS="${GPUS:-4}"
    NODE="${NODE:-master-02}"
    POD="${POD:-tinkercloud-miles}"
    CONTAINER="${CONTAINER:-tinkercloud-miles}"
    IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-gcp-secret}"
    RUN_SERVER=1 ;;   # Ray head + `python3 -m training` (TINKERCLOUD_BACKEND=miles)
  *) echo "unknown --profile $PROFILE (want: nemo_rl | bionemo | megatron_bridge | miles)"; exit 1 ;;
esac

# --- shared config (IMAGE/GPUS/NODE/POD/CONTAINER/BACKEND set per-profile) --
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-}"   # non-empty only for private-registry profiles (miles)
SERVER_LOG="${SERVER_LOG:-/data/server1.log}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
# k8s
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/ns.config}"
NS="${NS:-tinkercloud-nemorl}"
# docker
DATA_DIR="${DATA_DIR:-/opt/dlami/nvme/$USER}"
DOCKER="${DOCKER:-docker}"

# git-mode repos/refs
TINKER_CLOUD_REPO="${TINKER_CLOUD_REPO:-https://github.com/GMISWE/tinker-cloud.git}"
TINKER_CLOUD_REF="${TINKER_CLOUD_REF:-main}"
RL_REPO="${RL_REPO:-https://github.com/GavinZhu-GMI/RL.git}"
RL_REF="${RL_REF:-main}"
TINKER_GMI_REPO="${TINKER_GMI_REPO:-https://github.com/GMISWE/tinker_gmi.git}"
TINKER_GMI_REF="${TINKER_GMI_REF:-main}"
COOKBOOK_REPO="${COOKBOOK_REPO:-https://github.com/GMISWE/tinker-cookbook.git}"
COOKBOOK_REF="${COOKBOOK_REF:-main}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- transport abstraction: CP <local> <remote-path>, EX "<script>" --------
if [ "$TARGET" = k8s ]; then
  CP() { kubectl -n "$NS" cp "$1" "$POD:$2"; }
  EX() { kubectl -n "$NS" exec "$POD" -- bash -c "$1"; }
  EX_TIMEOUT() { timeout 30 kubectl -n "$NS" exec "$POD" -- bash -c "$1" || true; }
elif [ "$TARGET" = docker ]; then
  CP() { $DOCKER cp "$1" "$CONTAINER:$2"; }
  EX() { $DOCKER exec "$CONTAINER" bash -c "$1"; }
  EX_TIMEOUT() { timeout 30 $DOCKER exec "$CONTAINER" bash -c "$1" || true; }
else
  echo "unknown --target $TARGET"; exit 1
fi

# --- 1. create the pod / container -----------------------------------------
if [ "$CODE_ONLY" = 0 ]; then
  if [ "$TARGET" = k8s ]; then
    echo "==> [1/6] creating pod $POD on $NODE"
    # private-registry profiles (miles GAR image) need a pull secret; blank otherwise
    PULL_SECRET_YAML=""
    [ -n "$IMAGE_PULL_SECRET" ] && PULL_SECRET_YAML="imagePullSecrets: [{name: $IMAGE_PULL_SECRET}]"
    cat > "$TMP/pod.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  labels: {app: tinkercloud-nemorl}
  name: $POD
  namespace: $NS
spec:
  nodeName: $NODE          # bypasses scheduler; NoSchedule taints don't block
  restartPolicy: Never
  runtimeClassName: nvidia # MANDATORY: without it vLLM dies (no NVML)
  $PULL_SECRET_YAML
  tolerations:
  - {key: node.kubernetes.io/unschedulable, operator: Exists, effect: NoSchedule}
  containers:
  - name: tinkercloud
    image: $IMAGE
    command: [sleep, infinity]
    env:
    - {name: PYTHONUNBUFFERED, value: "1"}
    - {name: TOKENIZERS_PARALLELISM, value: "false"}
    - {name: TINKER_API_KEY, value: tml-dev-key}
    - {name: TINKERCLOUD_BACKEND, value: $BACKEND}
    - {name: ALLOW_PARTIAL_BATCHES, value: "true"}
    - {name: HF_HOME, value: /data/.cache/huggingface}
    - {name: CUDA_DEVICE_MAX_CONNECTIONS, value: "1"}
    - {name: NCCL_NVLS_ENABLE, value: "0"}
    - {name: NCCL_IGNORE_DISABLED_P2P, value: "1"}
    resources:
      requests: {cpu: "16", memory: 128Gi, nvidia.com/gpu: $GPUS}
      limits: {cpu: "48", memory: 256Gi, nvidia.com/gpu: $GPUS}
    volumeMounts:
    - {mountPath: /dev/shm, name: shm}
    - {mountPath: /data, name: data}
  volumes:
  - name: shm
    emptyDir: {medium: Memory, sizeLimit: 32Gi}
  - name: data
    emptyDir: {}
EOF
    kubectl apply -f "$TMP/pod.yaml"
    echo "==> waiting for Ready (image pull can take ~5-10 min on a cold node)"
    # NB: GPUS must be <= the node's UNALLOCATED nvidia.com/gpu (idle pods still
    # hold their allocation); otherwise kubelet rejects with UnexpectedAdmissionError.
    for _ in $(seq 1 240); do
      PHASE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}')
      READY=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
      [ "$READY" = True ] && break
      if [ "$PHASE" = Failed ]; then
        echo "ERROR: pod admission failed:"; kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.message}'; echo
        exit 1
      fi
      sleep 5
    done
    [ "$READY" = True ] || { echo "ERROR: pod not Ready after 20 min"; exit 1; }
  else
    echo "==> [1/6] starting container $CONTAINER (data: $DATA_DIR)"
    $DOCKER run -d --name "$CONTAINER" --init --gpus all --network host \
      --shm-size=32g -v "$DATA_DIR:/data" \
      -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
      -e TINKER_API_KEY=tml-dev-key -e TINKERCLOUD_BACKEND=$BACKEND \
      -e ALLOW_PARTIAL_BATCHES=true -e HF_HOME=/data/.cache/huggingface \
      -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e NCCL_NVLS_ENABLE=0 \
      -e NCCL_IGNORE_DISABLED_P2P=1 \
      "$IMAGE" sleep infinity
  fi
else
  echo "==> [1/6] --code-only: restarting server on existing $TARGET target"
  EX "pkill -f 'python3 -m trainin[g]' || true; sleep 8" || true
fi

# the megatron_bridge profile also bundles the bionemo-recipes evo2 recipe (built
# in-container). dev/monorepo only (submodule); git mode not supported for it.
EXTRA_TAR_PATHS=""
[ "$PROFILE" = megatron_bridge ] && EXTRA_TAR_PATHS="bionemo-recipes/recipes/evo2_megatron"

# --- 2. gather code ---------------------------------------------------------
echo "==> [2/6] gathering code (source=$SOURCE)"
if [ "$SOURCE" = dev ]; then
  # monorepo layout: tinker-cloud/scripts/ -> monorepo root is two dirs up
  MONOREPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
  for d in tinker-cloud RL tinker_gmi tinker-cookbook; do
    [ -d "$MONOREPO/$d" ] || { echo "ERROR: $MONOREPO/$d missing (dev mode needs the monorepo)"; exit 1; }
  done
  if [ -n "$EXTRA_TAR_PATHS" ] && [ ! -d "$MONOREPO/$EXTRA_TAR_PATHS" ]; then
    echo "ERROR: $MONOREPO/$EXTRA_TAR_PATHS missing (megatron_bridge needs the bionemo-recipes submodule: git submodule update --init bionemo-recipes)"; exit 1
  fi
  SRC="$MONOREPO"
elif [ "$SOURCE" = git ]; then
  [ -n "$EXTRA_TAR_PATHS" ] && { echo "ERROR: --profile megatron_bridge needs --source dev (bionemo-recipes submodule)"; exit 1; }
  echo "    cloning: tinker-cloud@$TINKER_CLOUD_REF RL@$RL_REF tinker_gmi@$TINKER_GMI_REF tinker-cookbook@$COOKBOOK_REF"
  git clone -q --depth 1 --branch "$TINKER_CLOUD_REF" "$TINKER_CLOUD_REPO" "$TMP/src/tinker-cloud"
  git clone -q --depth 1 --branch "$RL_REF"           "$RL_REPO"           "$TMP/src/RL"
  git clone -q --depth 1 --branch "$TINKER_GMI_REF"   "$TINKER_GMI_REPO"   "$TMP/src/tinker_gmi"
  git clone -q --depth 1 --branch "$COOKBOOK_REF"     "$COOKBOOK_REPO"     "$TMP/src/tinker-cookbook"
  SRC="$TMP/src"
else
  echo "unknown --source $SOURCE"; exit 1
fi
tar -C "$SRC" -czf "$TMP/code.tar.gz" \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='.venv' --exclude='tinker-cookbook/attic' \
  tinker-cloud/training tinker-cloud/tests tinker-cloud/scripts/evo2 \
  RL/nemo_rl \
  tinker_gmi/src tinker_gmi/pyproject.toml tinker_gmi/README.md \
  tinker-cookbook/tinker_cookbook tinker-cookbook/pyproject.toml tinker-cookbook/README.md \
  $EXTRA_TAR_PATHS
echo "    bundle: $(du -h "$TMP/code.tar.gz" | cut -f1)"

# --- 3. copy in --------------------------------------------------------------
echo "==> [3/6] copying code + token into target"
CP "$TMP/code.tar.gz" /tmp/code.tar.gz
CP "$SCRIPT_DIR/lib/setup_container.sh" /tmp/setup_container.sh
if [ -n "$HF_TOKEN_FILE" ]; then
  CP "$HF_TOKEN_FILE" /tmp/hf_token.txt
fi
if [ "$RUN_SERVER" = 1 ]; then
  EX "test -s /tmp/hf_token.txt" || {
    echo "ERROR: no HF token in target; pass HF_TOKEN_FILE=<path>"; exit 1; }
else
  # bionemo: the Evo2 checkpoint (arcinstitute) is public; token optional.
  EX "test -s /tmp/hf_token.txt" || echo "    note: no HF token (fine for public Evo2 weights)"
fi

# --- 4. in-container setup ---------------------------------------------------
echo "==> [4/6] running in-container setup (profile=$PROFILE)"
EX "PROFILE=$PROFILE bash /tmp/setup_container.sh"

if [ "$RUN_SERVER" = 0 ]; then
  # bionemo profile: env only — no Ray, no server (no working evo2 server yet).
  echo "==> DONE: faithful Evo2 env ready (profile=$PROFILE, target=$TARGET, image=$IMAGE)"
  echo "    code at /app/training; evo2 helpers at /app/scripts/evo2/"
  echo "    quickstart (inside the target):"
  echo "      export HF_HOME=/data/.cache/huggingface"
  echo "      evo2_convert_to_nemo2 --model-path hf://arcinstitute/savanna_evo2_1b_base \\"
  echo "        --model-size 1b --output-dir /data/evo2_1b_nemo2"
  echo "      python /app/scripts/evo2/splice_mkdata.py     # splice_sites_all -> fasta+labels"
  echo "      python /app/scripts/evo2/evo2_extract.py --fasta <fa> --ckpt-dir /data/evo2_1b_nemo2 --output-dir <out>"
  echo "    see specs/004-bionemo-classification/P5-RESULTS.md"
  exit 0
fi

# --- 5. ray head -------------------------------------------------------------
echo "==> [5/6] ensuring Ray head is up"
EX "ray status >/dev/null 2>&1 || { ray start --head --num-gpus $GPUS --disable-usage-stats > /tmp/ray_start.log 2>&1 < /dev/null; sleep 5; }
ray status 2>/dev/null | grep GPU || true"

# --- 6. server ---------------------------------------------------------------
echo "==> [6/6] starting API server (log: $SERVER_LOG)"
# NB: the detached child must have ALL fds redirected or the exec hangs (143);
# even then the launch usually succeeded — verified below with a fresh probe.
EX_TIMEOUT "
export PYTHONPATH=/app NUM_GPUS=$GPUS RAY_ADDRESS=ray://localhost:10001
export HF_TOKEN=\$(cat /tmp/hf_token.txt) HF_HOME=/data/.cache/huggingface \
       TINKER_API_KEY=tml-dev-key TINKERCLOUD_BACKEND=$BACKEND ALLOW_PARTIAL_BATCHES=true
cd /app && setsid nohup python3 -m training > $SERVER_LOG 2>&1 < /dev/null &
sleep 2; echo LAUNCHED"

CODE=000
for _ in $(seq 1 20); do
  CODE=$(EX "curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/health" || echo 000)
  [ "$CODE" = 200 ] && break
  sleep 5
done
if [ "$CODE" = 200 ]; then
  echo "==> DONE: TinkerCloud healthy (target=$TARGET, source=$SOURCE, log=$SERVER_LOG)"
else
  echo "==> FAILED health check; last server log lines:"
  EX "tail -20 $SERVER_LOG"
  exit 1
fi
