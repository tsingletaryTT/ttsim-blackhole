# ttsim-blackhole -- real ttnn code executed against a simulated Tenstorrent
# Blackhole chip, no Tenstorrent hardware anywhere in this container.
#
# Base image already ships a self-contained, pip-installed ttnn wheel (bundles
# its own kernel/runtime source), so no TT_METAL_HOME checkout or from-source
# tt-metal build is needed. All runtime shared libraries this stack needs
# (libmpi, libhwloc, libnuma, libevent-core/pthreads) are already present.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

# Build ttsim's Blackhole simulator library from source, pinned to v1.10.6
# (the latest tag as of 2026-09-09).
#
# History: v1.10.1 was pinned instead of latest for a while, because v1.10.3
# through v1.10.6 had a reproduced regression on Blackhole -- any
# ttnn.matmul with an output free dimension of 3072 (e.g. a GPT-2-style MLP
# up-projection, 768 -> 3072 -- exactly what the "Real HF Checkpoint" kernel
# needs) aborted with
# `UnsupportedFunctionality: tensix_pacr: Disable_pack_zero_flags`. On
# 2026-09-09, re-testing after the upstream `tt-metalium-*:latest-rc` base
# image moved (a routine tt-metal rebuild, unrelated to ttsim) found the bug
# no longer reproduces on EITHER v1.10.1 or v1.10.6 -- something changed on
# the tt-metal/ttnn side of the pairing, not in ttsim itself. Re-verified the
# full "Real HF Checkpoint" pipeline end-to-end on v1.10.6 before moving this
# pin. This is exactly the kind of thing that can un-fix itself on a future
# rebuild of either project -- if the wide-matmul case ever breaks again,
# that's why, and v1.10.1/v1.10.2 are the known-good fallback.
RUN git clone --depth=1 --branch v1.10.6 https://github.com/tenstorrent/ttsim.git /opt/ttsim-src \
    && cd /opt/ttsim-src \
    && ./make.py src/_out/release_bh/libttsim.so \
    && mkdir -p /opt/sim/bh \
    && cp src/_out/release_bh/libttsim.so /opt/sim/bh/libttsim_bh.so \
    && rm -rf /opt/ttsim-src

# ttsim resolves the SOC descriptor as a sibling of the .so file, named
# exactly `soc_descriptor.yaml`.
COPY soc_descriptor_bh.yaml /opt/sim/bh/soc_descriptor.yaml

# The 2-chip Blackhole (P300) simulator, for the mesh kernel -- prebuilt
# release binary (no source build needed for this one), same v1.10.6 pin.
RUN mkdir -p /opt/sim/bh_x2 \
    && curl -sL -o /opt/sim/bh_x2/libttsim_bh_x2.so \
        https://github.com/tenstorrent/ttsim/releases/download/v1.10.6/libttsim_bh_x2.so
COPY soc_descriptor_bh.yaml /opt/sim/bh_x2/soc_descriptor.yaml
COPY blackhole_P300_both_mmio.yaml /opt/sim/bh_x2/cluster_descriptor.yaml

# CPU-only torch + transformers (for real HF-checkpoint demos) and the API
# server's own deps, all into the same venv that already has ttnn -- one
# process, one interpreter, no dev-checkout PYTHONPATH tricks needed.
RUN uv pip install --python /opt/venv/bin/python3 \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /opt/venv/bin/python3 \
        transformers fastapi "uvicorn[standard]" websockets pydantic

COPY app /app
WORKDIR /app

ENV SIM_HOME=/opt/sim \
    TT_METAL_PYTHON=/opt/venv/bin/python3 \
    EXEC_TIMEOUT=180 \
    PORT=7860 \
    PATH=/opt/venv/bin:$PATH

EXPOSE 7860
CMD ["/opt/venv/bin/python3", "api_server.py"]
