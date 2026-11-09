#!/usr/bin/env bash
# One-key deploy of a TinkerCloud dev environment (k8s pod or docker container).
#
#   deploy_tinkercloud.sh [--profile nemo_rl|bionemo|megatron_bridge|miles|automodel] \
#                         [--source dev|git] [--target k8s|docker] [--code-only]
#
# Profiles (cased by BASE IMAGE — the images ship incompatible stacks)
#   nemo_rl  (default) NeMo RL server backend. Image nvcr.io/nvidia/nemo-rl:*.
#            Overlays RL/nemo_rl, starts a Ray head + `python3 -m training`
#            (TINKERCLOUD_BACKEND=nemo_rl), health-checks the API port (API_PORT). GPUS=4.
#   bionemo  cu12 NeMo2 exploration env (feature 004 P5). Image
#            nvcr.io/nvidia/clara/bionemo-framework:* — hyena Megatron-LM + TE +
#            bionemo.evo2 + NeMo2. Env only (code + scripts/evo2 helpers, no server).
#   megatron_bridge  cu13 recipe env (feature 004 P5, the FAITHFUL Evo2 classifier).
#            Image nvcr.io/nvidia/pytorch:26.04-py3. Builds the bionemo-recipes
#            evo2_megatron recipe (megatron-bridge v0.4.1) so the megatron_bridge
#            backend can import megatron.bridge + evo2_classifier. Env only until
#            that backend lands (stubs today). GPUS=2.
#   miles    Miles/Slime backend. Image is the private GMI GAR build of
#            github.com/GavinZhu-GMI/miles (miles + Megatron/SGLang/TE pre-installed);
#            overlays only the server code, starts Ray head + `python3 -m training`
#            (TINKERCLOUD_BACKEND=miles), health-checks the API port (API_PORT). GPUS=4.
#            Needs a pull secret for the private image: IMAGE_PULL_SECRET (default
#            gcp-secret) must exist in namespace $NS.
#   automodel  Automodel backend env (specs/027). Image built from
#            training/backends/automodel/docker: nemo-automodel training env in /opt/venv plus the
#            NGC vLLM env in /opt/vllm-env (the backend starts it per base model).
#            Private registry image (pull secret gar-secret). Installs the server,
#            SDK and cookbook into /opt/venv, starts Ray head + `python3 -m training`
#            (TINKERCLOUD_BACKEND=automodel), health-checks the API port (API_PORT). GPUS=4.
#            The backend's required knobs default here and are overridable:
#            AUTOMODEL_TRAIN_GPUS=2 AUTOMODEL_MAX_LORAS=4 AUTOMODEL_MAX_LORA_RANK=64
#            AUTOMODEL_MAX_CPU_LORAS=8 AUTOMODEL_MICRO_BATCH_ROWS=4 AUTOMODEL_ACTIVATION_CHECKPOINTING=0.
#   Per-profile defaults (IMAGE, GPUS, NODE, BACKEND) are overridable by env vars;
#   KUBECONFIG, NS and the node come from scripts/lib/cluster.sh.
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
# HF token: HF_TOKEN_FILE=<path>, copied into the target as HF_TOKEN_POD_FILE (lib/cluster.sh).
# Unset, it is huggingface_hub's own token file when one exists; otherwise the
# target's existing HF_TOKEN_POD_FILE is reused (--code-only only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Variables the caller set, before any profile default fills one in (--code-only, below).
CALLER_ENV=" $(compgen -e | tr '\n' ' ') "
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
# shellcheck source=scripts/lib/cluster.sh
# shellcheck disable=SC1091  # sourced by path at run time: KUBECONFIG, NS, DEFAULT_NODE
. "$SCRIPT_DIR/lib/cluster.sh"   # KUBECONFIG, NS, kx, EX, EX_TIMEOUT, pod_cmd, ray_head, TRUST_DOMAIN_LABEL, SERVER_PATTERN, SERVER_STOP, API_PORT, HF_TOKEN_POD_FILE, HF_TOKEN_CHECK, server_boot, SERVER_BOOT_TIMEOUT, SERVER_HEALTH
BACKEND_ENV=()   # backend knobs as KEY=VAL, exported to the server process (step 6)
case "$PROFILE" in
  nemo_rl)
    IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.5.0}"
    BACKEND="${BACKEND:-nemo_rl}"
    GPUS="${GPUS:-4}"
    NODE="${NODE:-$DEFAULT_NODE}"
    POD="${POD:-tinkercloud-nemorl}"
    CONTAINER="${CONTAINER:-tinkercloud-nemorl}"
    RUN_SERVER=1 ;;   # Ray head + `python3 -m training` + health check
  bionemo)
    IMAGE="${IMAGE:-nvcr.io/nvidia/clara/bionemo-framework:2.7.1}"
    BACKEND="${BACKEND:-automodel}"
    GPUS="${GPUS:-2}"
    NODE="${NODE:-$DEFAULT_NODE}"
    POD="${POD:-bionemo-evo2-p5}"
    CONTAINER="${CONTAINER:-bionemo-evo2-p5}"
    RUN_SERVER=0 ;;   # env only (cu12 NeMo2 exploration env): code + evo2 helpers
  megatron_bridge)
    # cu13 recipe env: builds bionemo-recipes evo2_megatron (megatron-bridge) so
    # the megatron_bridge backend can import megatron.bridge + the Evo2 classifier.
    IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
    BACKEND="${BACKEND:-megatron_bridge}"
    GPUS="${GPUS:-2}"
    NODE="${NODE:-$DEFAULT_NODE}"
    POD="${POD:-evo2-recipe}"
    CONTAINER="${CONTAINER:-evo2-recipe}"
    RUN_SERVER=0 ;;   # env only until the megatron_bridge backend lands (stubs today)
  miles)
    # Miles/Slime backend. Miles + its Megatron/SGLang/TE stack are pre-installed
    # in the GMI base image (built from github.com/GavinZhu-GMI/miles); we overlay
    # only the tinker-cloud server code. The image is a PRIVATE GAR repo, so the
    # pod needs an imagePullSecret (IMAGE_PULL_SECRET, default gcp-secret).
    IMAGE="${IMAGE:-us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/miles_tinker_seam:20260730a}"
    BACKEND="${BACKEND:-miles}"
    GPUS="${GPUS:-4}"
    NODE="${NODE:-$DEFAULT_NODE}"
    POD="${POD:-tinkercloud-miles}"
    CONTAINER="${CONTAINER:-tinkercloud-miles}"
    IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-gcp-secret}"
    RUN_SERVER=1 ;;   # Ray head + `python3 -m training` (TINKERCLOUD_BACKEND=miles)
  automodel)
    # Built from training/backends/automodel/docker. Training runs from /opt/venv, vLLM from
    # /opt/vllm-env; the deploy installs pure-Python packages into the first
    # only (setup_container.sh) and never touches the second.
    IMAGE="${IMAGE:-us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/tinkercloud-automodel:26.08.00-vllm26.09-plugin}"
    IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-gar-secret}"
    BACKEND="${BACKEND:-automodel}"
    GPUS="${GPUS:-4}"
    NODE="${NODE:-$DEFAULT_NODE}"
    POD="${POD:-tinkercloud-automodel}"
    CONTAINER="${CONTAINER:-tinkercloud-automodel}"
    # AutomodelConfig has no defaults (a server without them fails at startup);
    # these are the deploy's. TRAIN_GPUS ranks per training group, the rest of
    # GPUS serve one vLLM per base model. MAX_LORAS: adapters resident on a
    # vLLM's GPU; a tenant holds two across an optimizer step (D20), so 4 is
    # two tenants. MAX_LORA_RANK: the largest LoRA rank create_model accepts.
    BACKEND_ENV=("AUTOMODEL_TRAIN_GPUS=${AUTOMODEL_TRAIN_GPUS:-2}" "AUTOMODEL_MAX_LORAS=${AUTOMODEL_MAX_LORAS:-4}"
      "AUTOMODEL_MAX_LORA_RANK=${AUTOMODEL_MAX_LORA_RANK:-64}" "AUTOMODEL_MAX_CPU_LORAS=${AUTOMODEL_MAX_CPU_LORAS:-8}"
      "AUTOMODEL_MICRO_BATCH_ROWS=${AUTOMODEL_MICRO_BATCH_ROWS:-4}"
      "AUTOMODEL_ACTIVATION_CHECKPOINTING=${AUTOMODEL_ACTIVATION_CHECKPOINTING:-0}")
    RUN_SERVER=1 ;;   # Ray head + `python3 -m training` (TINKERCLOUD_BACKEND=automodel)
  *) echo "unknown --profile $PROFILE (want: nemo_rl | bionemo | megatron_bridge | miles | automodel)"; exit 1 ;;
esac

# --- shared config (IMAGE/GPUS/NODE/POD/CONTAINER/BACKEND set per-profile) --
IMAGE_PULL_SECRET="${IMAGE_PULL_SECRET:-}"   # non-empty only for private-registry profiles (miles)
SERVER_LOG="${SERVER_LOG:-/data/server1.log}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
[ -z "$HF_TOKEN_FILE" ] && [ -s "$HOME/.cache/huggingface/token" ] && HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
# k8s: a host directory in HF_HOME layout (hub/models--...) mounted at
# /data/.cache/huggingface, so weights outlive the pod and are shared by
# every pod on the node. Blank keeps the cache in the pod's emptyDir.
HF_CACHE_HOSTPATH="${HF_CACHE_HOSTPATH:-}"
# k8s: size of a PersistentVolumeClaim "<pod>-data" for /data (the cluster's
# default StorageClass), so checkpoints, metadata and logs outlive the pod.
# Blank keeps /data an emptyDir. The claim is left in place when the pod goes.
DATA_PVC_SIZE="${DATA_PVC_SIZE:-}"
POD_MEMORY_LIMIT="${POD_MEMORY_LIMIT:-256Gi}"   # host-RAM ceiling of the pod (k8s target)
# k8s
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
  require_tunnel
  CP() { kubectl -n "$NS" cp "$1" "$POD:$2"; }
  # EX, EX_TIMEOUT: lib/cluster.sh's
  # Capabilities the pod spec adds. Kubernetes fixes them at pod creation and
  # `kubectl apply` changes only the image, so an older pod keeps its old set.
  POD_CAPS="SYS_PTRACE"
  check_pod_caps() {
    local live cap pvc gpus hf image mem
    live=$(kx get pod "$POD" -o jsonpath='{.spec.containers[0].securityContext.capabilities.add}') \
      || { echo "ERROR: could not read pod $POD's capabilities"; exit 1; }
    for cap in $POD_CAPS; do
      case "$live" in *"\"$cap\""*) ;; *)
        echo "ERROR: pod $POD lacks $cap, which the pod spec adds (live: ${live:-none})"
        # the recreate repeats the live pod's GPUs, image, /data PVC and HF cache:
        # without DATA_PVC_SIZE the new pod mounts an empty emptyDir at /data
        gpus=$(kx get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}')
        image=$(kx get pod "$POD" -o jsonpath='{.spec.containers[0].image}')
        mem=$(kx get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.memory}')
        hf=$(kx get pod "$POD" -o jsonpath='{.spec.volumes[?(@.name=="hfcache")].hostPath.path}')
        pvc=$(kx get pod "$POD" -o jsonpath='{.spec.volumes[?(@.name=="data")].persistentVolumeClaim.claimName}')
        [ -n "$pvc" ] && pvc=$(kx get pvc "$pvc" -o jsonpath='{.spec.resources.requests.storage}')
        echo "  recreate it: kubectl -n $NS delete pod $POD && POD=$POD GPUS=$gpus IMAGE=$image POD_MEMORY_LIMIT=$mem" \
             "${pvc:+DATA_PVC_SIZE=$pvc }${hf:+HF_CACHE_HOSTPATH=$hf }$0 --profile $PROFILE"
        [ -n "$pvc" ] || echo "  (its /data is an emptyDir: the recreate loses it)"
        exit 1 ;;
      esac
    done
  }
elif [ "$TARGET" = docker ]; then
  # shellcheck disable=SC2086  # DOCKER may carry words ("sg docker -c")
  CP() { $DOCKER cp "$1" "$CONTAINER:$2"; }
  # shellcheck disable=SC2086
  EX() { $DOCKER exec "$CONTAINER" bash -c "$(pod_cmd "$1")"; }
  # shellcheck disable=SC2086
  EX_TIMEOUT() { timeout "$1" $DOCKER exec "$CONTAINER" bash -c "$(pod_cmd "$2")" || true; }
else
  echo "unknown --target $TARGET"; exit 1
fi

# A new target has no token to reuse: refuse before anything is created.
if [ "$CODE_ONLY" = 0 ] && [ "$RUN_SERVER" = 1 ] && [ ! -s "$HF_TOKEN_FILE" ]; then
  echo "ERROR: a new $PROFILE target needs an HF token: HF_TOKEN_FILE='${HF_TOKEN_FILE}' is missing or empty"
  exit 1
fi

# --code-only keeps the running server's layout: NUM_GPUS and every BACKEND_ENV knob the
# caller did not set take the live server's value, not the profile default (a default
# AUTOMODEL_TRAIN_GPUS=2 on an 8-GPU server silently halves the training group).
if [ "$CODE_ONLY" = 1 ] && [ "$RUN_SERVER" = 1 ]; then
  live_env=$(EX "for p in \$(pgrep -f '$SERVER_PATTERN'); do tr '\0' '\n' < /proc/\$p/environ; done") \
    || { echo "ERROR: could not read the running server's environment on $TARGET"; exit 1; }
  if [ -z "$live_env" ]; then
    echo "==> --code-only: no server running; using the profile's layout"
  else
    live_of() { printf '%s\n' "$live_env" | sed -n "s/^$1=//p" | head -1; }
    keep() {  # $1 caller's variable name, $2 key in the server env, $3 value this deploy would use
      local live; live=$(live_of "$2")
      if [ -z "$live" ] || [ "$live" = "$3" ]; then printf '%s' "$3"
      elif [[ "$CALLER_ENV" == *" $1 "* ]]; then echo "==> --code-only: $2 $live -> $3 (set by the caller)" >&2; printf '%s' "$3"
      else echo "==> --code-only: keeping the running server's $2=$live (profile default $3)" >&2; printf '%s' "$live"
      fi
    }
    GPUS=$(keep GPUS NUM_GPUS "$GPUS")
    kept=()
    for kv in "${BACKEND_ENV[@]}"; do
      kept+=("${kv%%=*}=$(keep "${kv%%=*}" "${kv%%=*}" "${kv#*=}")")
    done
    BACKEND_ENV=("${kept[@]}")
  fi
fi

# --- 1. create the pod / container -----------------------------------------
if [ "$CODE_ONLY" = 0 ]; then
  if [ "$TARGET" = k8s ]; then
    echo "==> [1/6] creating pod $POD on $NODE"
    # private-registry profiles (miles GAR image) need a pull secret; blank otherwise
    PULL_SECRET_YAML=""
    [ -n "$IMAGE_PULL_SECRET" ] && PULL_SECRET_YAML="imagePullSecrets: [{name: $IMAGE_PULL_SECRET}]"
    HF_MOUNT_YAML="" HF_VOLUME_YAML=""
    if [ -n "$HF_CACHE_HOSTPATH" ]; then
      HF_MOUNT_YAML="- {mountPath: /data/.cache/huggingface, name: hfcache}"
      HF_VOLUME_YAML="- {name: hfcache, hostPath: {path: $HF_CACHE_HOSTPATH, type: Directory}}"
    fi
    DATA_VOLUME_YAML="emptyDir: {}"
    if [ -n "$DATA_PVC_SIZE" ]; then
      DATA_VOLUME_YAML="persistentVolumeClaim: {claimName: $POD-data}"
      # The pod sets nodeName, so the scheduler never picks a node for the
      # claim; a WaitForFirstConsumer class (local-path) then never provisions
      # it. The annotation is what the scheduler would have written.
      cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $POD-data
  namespace: $NS
  annotations: {volume.kubernetes.io/selected-node: $NODE}
spec:
  accessModes: [ReadWriteOnce]
  resources: {requests: {storage: $DATA_PVC_SIZE}}
EOF
    fi
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
    # py-spy dump / ray stack attach to non-child processes (Ray workers);
    # the runtime's default capability set omits it and the host's Yama
    # ptrace_scope=1 then refuses the attach (observed: py-spy dump on a Ray
    # worker fails with EACCES)
    securityContext: {capabilities: {add: [${POD_CAPS// /, }]}}
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
      limits: {cpu: "48", memory: $POD_MEMORY_LIMIT, nvidia.com/gpu: $GPUS}
    volumeMounts:
    - {mountPath: /dev/shm, name: shm}
    - {mountPath: /data, name: data}
    $HF_MOUNT_YAML
  volumes:
  - name: shm
    emptyDir: {medium: Memory, sizeLimit: 32Gi}
  - name: data
    $DATA_VOLUME_YAML
  $HF_VOLUME_YAML
EOF
    [ -n "$(kx get pod "$POD" --ignore-not-found -o name)" ] && check_pod_caps
    kubectl apply -f "$TMP/pod.yaml"
    echo "==> waiting for Ready on $IMAGE (image pull can take ~5-10 min on a cold node)"
    # NB: GPUS must be <= the node's UNALLOCATED nvidia.com/gpu (idle pods still
    # hold their allocation); otherwise kubelet rejects with UnexpectedAdmissionError.
    # On an existing pod, apply changes only the image: kubelet restarts the
    # container in place (/data survives) while the old one still reports
    # Ready, so Ready counts only once the running container has the image.
    for _ in $(seq 1 240); do
      PHASE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}')
      READY=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
      RUNNING_IMAGE=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].image}')
      # the runtime may report a short name registry-qualified
      case "$RUNNING_IMAGE" in "$IMAGE"|*/"$IMAGE") ;; *) READY=False ;; esac
      [ "$READY" = True ] && break
      if [ "$PHASE" = Failed ]; then
        echo "ERROR: pod admission failed:"; kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.message}'; echo
        exit 1
      fi
      sleep 5
    done
    [ "$READY" = True ] || { echo "ERROR: pod not Ready on $IMAGE after 20 min (running: $RUNNING_IMAGE)"; exit 1; }
  else
    echo "==> [1/6] starting container $CONTAINER (data: $DATA_DIR)"
    # only the API port leaves the container: Ray's ports and the vLLM servers stay inside it
    $DOCKER run -d --name "$CONTAINER" --init --gpus all -p "$API_PORT:$API_PORT" \
      --shm-size=32g -v "$DATA_DIR:/data" \
      -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false \
      -e TINKER_API_KEY=tml-dev-key -e TINKERCLOUD_BACKEND="$BACKEND" \
      -e ALLOW_PARTIAL_BATCHES=true -e HF_HOME=/data/.cache/huggingface \
      -e CUDA_DEVICE_MAX_CONNECTIONS=1 -e NCCL_NVLS_ENABLE=0 \
      -e NCCL_IGNORE_DISABLED_P2P=1 \
      "$IMAGE" sleep infinity
  fi
else
  echo "==> [1/6] --code-only: restarting server on existing $TARGET target"
  [ "$TARGET" = k8s ] && check_pod_caps
fi

# The policy admits, on every port, the pods of this pod's trust domain (Ray, NCCL bootstrap,
# vLLM), and anyone else on the API port only; egress stays open (model downloads). It takes
# effect only where the cluster's CNI enforces NetworkPolicy. Every deploy labels the pod and
# applies the policy, so a pod created before either existed gets both.
if [ "$TARGET" = k8s ]; then
  kx label pod "$POD" "$TRUST_DOMAIN_LABEL=$POD" --overwrite > /dev/null
  cat > "$TMP/netpol.yaml" <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: $POD-trust-domain
  namespace: $NS
spec:
  podSelector: {matchLabels: {$TRUST_DOMAIN_LABEL: $POD}}
  policyTypes: [Ingress]
  ingress:
  - from: [{podSelector: {matchLabels: {$TRUST_DOMAIN_LABEL: $POD}}}]
  - ports: [{port: $API_PORT, protocol: TCP}]
EOF
  kx apply -f "$TMP/netpol.yaml" > /dev/null
  echo "    NetworkPolicy $POD-trust-domain: pods labelled $TRUST_DOMAIN_LABEL=$POD on every port, anyone on :$API_PORT (if the CNI enforces it)"
fi

# the megatron_bridge profile also bundles the bionemo-recipes evo2 recipe (built
# in-container). dev/monorepo only (submodule); git mode not supported for it.
EXTRA_TAR_PATHS=""
[ "$PROFILE" = megatron_bridge ] && EXTRA_TAR_PATHS="bionemo-recipes/recipes/evo2_megatron"
# nemo_rl dev deploys carry the RL checkout's Megatron-Bridge package (its PEFT
# carries); setup_container overlays it on the image's. A git deploy carries none.
[ "$PROFILE" = nemo_rl ] && [ "$SOURCE" = dev ] && \
  EXTRA_TAR_PATHS="RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge"

# --- 2. gather code ---------------------------------------------------------
echo "==> [2/6] gathering code (source=$SOURCE)"
if [ "$SOURCE" = dev ]; then
  # monorepo layout: tinker-cloud/scripts/ -> monorepo root is two dirs up
  MONOREPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
  for d in tinker-cloud RL tinker_gmi tinker-cookbook; do
    [ -d "$MONOREPO/$d" ] || { echo "ERROR: $MONOREPO/$d missing (dev mode needs the monorepo)"; exit 1; }
  done
  if [ -n "$EXTRA_TAR_PATHS" ] && [ ! -d "$MONOREPO/$EXTRA_TAR_PATHS" ]; then
    echo "ERROR: $MONOREPO/$EXTRA_TAR_PATHS missing (a submodule not checked out: git submodule update --init --recursive)"; exit 1
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
  tinker-cloud/training tinker-cloud/tests tinker-cloud/gates tinker-cloud/pyproject.toml \
  tinker-cloud/scripts/evo2 tinker-cloud/scripts/automodel tinker-cloud/scripts/gates \
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
  CP "$HF_TOKEN_FILE" "$HF_TOKEN_POD_FILE"
fi
if [ "$RUN_SERVER" = 1 ]; then
  EX "$HF_TOKEN_CHECK" || {
    echo "ERROR: no HF token in target; pass HF_TOKEN_FILE=<path>"; exit 1; }
else
  # bionemo: the Evo2 checkpoint (arcinstitute) is public; token optional.
  EX "$HF_TOKEN_CHECK" > /dev/null || echo "    note: no HF token (fine for public weights)"
fi

# --- 4. in-container setup ---------------------------------------------------
# A running server stops here, once: after every step that can fail without touching the target,
# before setup replaces its code and packages and step 5 its Ray head. A RUN_SERVER=0 profile runs
# no server.
if [ "$RUN_SERVER" = 1 ]; then
  # refuse bad input while the running server still serves
  BOOT_CMD=$(server_boot "$GPUS" "$BACKEND" "$SERVER_LOG" "${BACKEND_ENV[@]}")
  EX "$SERVER_STOP"
fi
echo "==> [4/6] running in-container setup (profile=$PROFILE)"
EX "PROFILE=$PROFILE MILES_REPO='${MILES_REPO:-}' MILES_REF='${MILES_REF:-}' bash /tmp/setup_container.sh"

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
EX "$(ray_head ensure "$GPUS")"
EX "ray status 2>/dev/null | grep GPU || true"

# --- 6. server ---------------------------------------------------------------
echo "==> [6/6] starting API server (log: $SERVER_LOG)"
# the exec's exit is no verdict (server_boot); the health probe below is
EX_TIMEOUT "$SERVER_BOOT_TIMEOUT" "$BOOT_CMD"

CODE=000
for _ in $(seq 1 20); do
  CODE=$(EX "$SERVER_HEALTH") || true
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
