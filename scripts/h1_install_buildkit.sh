#!/usr/bin/env bash
# Install BuildKit + nerdctl on an RKE2 node, attached to RKE2's own containerd, so an image
# built here lands in the k8s.io namespace kubelet pulls from (no registry, no import).
# Installed into its own directory: RKE2's containerd/ctr/crictl in /usr/local/bin are untouched.
# Usage (on the node): sudo bash h1_install_buildkit.sh
set -euo pipefail
V=2.4.1
DIR=/opt/nerdctl-full-${V}
SOCK=/run/k3s/containerd/containerd.sock
STATE=/mnt/fast-disks/buildkit
TAR=nerdctl-full-${V}-linux-amd64.tar.gz
URL=https://github.com/containerd/nerdctl/releases/download/v${V}

[ -S "$SOCK" ] || { echo "no containerd socket at $SOCK" >&2; exit 1; }
work=$(mktemp -d)
cd "$work"
curl -fsSLO "${URL}/${TAR}"
curl -fsSLO "${URL}/SHA256SUMS"
grep " ${TAR}\$" SHA256SUMS | sha256sum -c -

mkdir -p "$DIR" "$STATE" /etc/buildkit /etc/nerdctl
tar -C "$DIR" -xzf "$TAR"
for b in nerdctl buildctl buildkitd; do ln -sf "$DIR/bin/$b" "/usr/local/bin/$b"; done

cat > /etc/buildkit/buildkitd.toml <<EOF
root = "${STATE}"
[worker.oci]
  enabled = false
[worker.containerd]
  enabled = true
  address = "${SOCK}"
  namespace = "k8s.io"
  gc = true
  gckeepstorage = "500GB"
EOF

cat > /etc/nerdctl/nerdctl.toml <<EOF
address = "unix://${SOCK}"
namespace = "k8s.io"
EOF

cat > /etc/systemd/system/buildkit.service <<EOF
[Unit]
Description=BuildKit (containerd worker on RKE2's containerd)
After=rke2-server.service
[Service]
ExecStart=${DIR}/bin/buildkitd --config /etc/buildkit/buildkitd.toml
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now buildkit.service
sleep 2
buildctl debug workers
nerdctl version | head -3
rm -rf "$work"
