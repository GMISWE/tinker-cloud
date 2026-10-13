# TinkerCloud automodel image: two vendor-built envs in one image.
#   /opt/venv       training env (Automodel, Ray, TE), from the base image, unchanged
#   /opt/vllm-env   sampling env (vLLM), copied whole from the NGC vLLM image
# Both NGC images ship the same torch build, so neither env is rebuilt.
# Start vLLM only through /opt/vllm-env/bin/vllm-api-server: it applies the
# vLLM image's environment to that process and to no other process.
#
#   docker build -t <repo>/tinkercloud-automodel:<tag> docker/automodel
ARG AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.08.00
ARG VLLM_IMAGE=nvcr.io/nvidia/vllm:26.06-py3

# Shared libraries that exist only in the vLLM image (by full path), flattened
# into one directory. The vLLM env loads them via LD_LIBRARY_PATH.
FROM ${AUTOMODEL_IMAGE} AS base_libs
RUN find /usr/local/lib /usr/lib/x86_64-linux-gnu -maxdepth 1 \( -type f -o -type l \) -name '*.so*' \
      | LC_ALL=C sort -u > /base_libs.txt

FROM ${VLLM_IMAGE} AS vllm_libs
COPY --from=base_libs /base_libs.txt /base_libs.txt
RUN find /usr/local/lib /usr/lib/x86_64-linux-gnu -maxdepth 1 \( -type f -o -type l \) -name '*.so*' \
      | LC_ALL=C sort -u > /vllm_libs.txt \
 && LC_ALL=C comm -23 /vllm_libs.txt /base_libs.txt > /only_vllm.txt \
 && mkdir /out \
 && tar -cf - --transform 's#.*/##' -T /only_vllm.txt | tar -xf - -C /out \
 && echo "vllm-only libs: $(ls /out | wc -l)"

FROM ${VLLM_IMAGE} AS vllm

FROM ${AUTOMODEL_IMAGE}
RUN /usr/bin/python3.12 -m venv --without-pip /opt/vllm-env
COPY --from=vllm /usr/local/lib/python3.12/dist-packages /opt/vllm-env/lib/python3.12/site-packages
# opencv's bundled libs link to these paths by absolute symlink
COPY --from=vllm /opt/ffmpeg-safe /opt/ffmpeg-safe
COPY --from=vllm /opt/ffmpeg-safe-cv2-shim-v5 /opt/ffmpeg-safe-cv2-shim-v5
COPY --from=vllm /opt/ffmpeg-safe-cv2-shim-v8 /opt/ffmpeg-safe-cv2-shim-v8
COPY --from=vllm_libs /out /opt/vllm-syslib
COPY vllm-api-server /opt/vllm-env/bin/vllm-api-server
RUN chmod 755 /opt/vllm-env/bin/vllm-api-server \
 && /opt/vllm-env/bin/vllm-api-server --check
