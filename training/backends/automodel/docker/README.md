# automodel image

Built on h1 with BuildKit attached to RKE2's containerd, so the image lands where kubelet pulls from
(`k8s.io` namespace, no registry):

```bash
sudo nerdctl build -f training/backends/automodel/docker/Dockerfile -t <name>:<tag> .   # from the tinker-cloud root
```

BuildKit is installed once per node by `scripts/h1_install_buildkit.sh`. A pod uses the image with
`IMAGE=<name>:<tag>` on `scripts/deploy_tinkercloud.sh --profile automodel` (IfNotPresent: no pull).
