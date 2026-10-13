# TinkerCloud automodel image: two vendor-built envs in one image.
#   /opt/venv       training env (Automodel, Ray, TE), from the base image, unchanged
#   /opt/vllm-env   sampling env (vLLM), copied whole from the NGC vLLM image
# The vLLM image is on a newer CUDA than the base, so the vLLM env also gets
# that image's CUDA toolkit (/usr/local/cuda-13.4, with its forward-compat
# driver), HPC-X (/opt/vllm-hpcx) and every system library whose content
# differs from the base (/opt/vllm-syslib). The base's /usr/local/cuda is
# untouched. Neither env is rebuilt.
# Start vLLM only through /opt/vllm-env/bin/vllm-api-server: it applies the
# vLLM image's environment to that process and to no other process.
#
#   docker build -t <repo>/tinkercloud-automodel:<tag> docker/automodel
ARG AUTOMODEL_IMAGE=nvcr.io/nvidia/nemo-automodel:26.08.00
ARG VLLM_IMAGE=nvcr.io/nvidia/vllm:26.09-py3

FROM ${AUTOMODEL_IMAGE} AS base_libs
COPY libs-manifest.sh /libs-manifest.sh
RUN sh /libs-manifest.sh > /base_libs.txt

# System libraries the base lacks or ships with different content, flattened
# into one directory. glibc is excluded: it must match the base's ld.so.
FROM ${VLLM_IMAGE} AS vllm_libs
COPY libs-manifest.sh /libs-manifest.sh
COPY --from=base_libs /base_libs.txt /base_libs.txt
RUN sh /libs-manifest.sh > /vllm_libs.txt \
 && LC_ALL=C comm -23 /vllm_libs.txt /base_libs.txt | cut -d' ' -f1 \
    | grep -Ev '/(ld-linux[^/]*|lib(c|m|dl|pthread|rt|util|mvec|resolv|anl|nsl|nss_[a-z]+|thread_db|BrokenLocale|c_malloc_debug|memusage|pcprofile))\.so[^/]*$' \
    > /only_vllm.txt \
 && mkdir /out \
 && tar -cf - --transform 's#.*/##' -T /only_vllm.txt | tar -xf - -C /out \
 && dangling=$(find /out -xtype l) \
 && { [ -z "$dangling" ] || { echo "dangling links in vllm-syslib: $dangling"; exit 1; }; } \
 && echo "vllm-only libs: $(ls /out | wc -l)"

FROM ${VLLM_IMAGE} AS vllm

FROM ${AUTOMODEL_IMAGE}
RUN /usr/bin/python3.12 -m venv --without-pip /opt/vllm-env
COPY --from=vllm /usr/local/lib/python3.12/dist-packages /opt/vllm-env/lib/python3.12/site-packages
COPY --from=vllm /usr/local/cuda-13.4 /usr/local/cuda-13.4
COPY --from=vllm /opt/hpcx /opt/vllm-hpcx
# opencv's bundled libs link to these paths by absolute symlink
COPY --from=vllm /opt/ffmpeg-safe /opt/ffmpeg-safe
COPY --from=vllm /opt/ffmpeg-safe-cv2-shim-v5 /opt/ffmpeg-safe-cv2-shim-v5
COPY --from=vllm /opt/ffmpeg-safe-cv2-shim-v8 /opt/ffmpeg-safe-cv2-shim-v8
COPY --from=vllm_libs /out /opt/vllm-syslib
COPY vllm-api-server /opt/vllm-env/bin/vllm-api-server
RUN chmod 755 /opt/vllm-env/bin/vllm-api-server \
 && test -x /usr/local/bin/cudaCheck \
 && /opt/vllm-env/bin/vllm-api-server --check
