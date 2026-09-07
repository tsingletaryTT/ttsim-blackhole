# ttsim-blackhole -- real ttnn code executed against a simulated Tenstorrent
# Blackhole chip, no Tenstorrent hardware anywhere in this container.
#
# Base image already ships a self-contained, pip-installed ttnn wheel (bundles
# its own kernel/runtime source), so no TT_METAL_HOME checkout or from-source
# tt-metal build is needed. All runtime shared libraries this stack needs
# (libmpi, libhwloc, libnuma, libevent-core/pthreads) are already present.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

# Build ttsim's Blackhole simulator library from source, pinned to v1.10.1.
# NOT the latest tag: bisected every v1.10.x release through v1.10.6
# (2026-09-07, latest at the time) -- v1.10.1 and v1.10.2 are clean, v1.10.3
# onward all have a regression on Blackhole: any ttnn.matmul with an output
# free dimension of 3072 (e.g. a GPT-2-style MLP up-projection, 768 -> 3072 --
# exactly what the "Real HF Checkpoint" kernel needs) aborts with
# `UnsupportedFunctionality: tensix_pacr: Disable_pack_zero_flags`, even
# though the same op sequence at narrower widths (768) is fine. No measurable
# perf difference on the ops this Space actually uses (attention, mesh)
# between v1.10.1 and v1.10.6 either, so there is currently no upgrade
# benefit that doesn't break the "Real HF Checkpoint" kernel. Re-test before
# moving this pin.
RUN git clone --depth=1 --branch v1.10.1 https://github.com/tenstorrent/ttsim.git /opt/ttsim-src \
    && cd /opt/ttsim-src \
    && ./make.py src/_out/release_bh/libttsim.so \
    && mkdir -p /opt/sim/bh \
    && cp src/_out/release_bh/libttsim.so /opt/sim/bh/libttsim_bh.so \
    && rm -rf /opt/ttsim-src

# ttsim resolves the SOC descriptor as a sibling of the .so file, named
# exactly `soc_descriptor.yaml`.
COPY soc_descriptor_bh.yaml /opt/sim/bh/soc_descriptor.yaml

# A second build at the latest tag (the regression described above, still
# reproduced as of this version), used ONLY by the "Break the Rules"
# playground kernel to show a real, still-open Blackhole limitation live --
# every other kernel uses the stable v1.10.1 pin above. Pinned to an exact
# tag (not a floating clone of the default branch) so this kernel's behavior
# doesn't silently change on an unrelated rebuild.
RUN git clone --depth=1 --branch v1.10.6 https://github.com/tenstorrent/ttsim.git /opt/ttsim-src-head \
    && cd /opt/ttsim-src-head \
    && ./make.py src/_out/release_bh/libttsim.so \
    && mkdir -p /opt/sim/bh_head \
    && cp src/_out/release_bh/libttsim.so /opt/sim/bh_head/libttsim_bh.so \
    && rm -rf /opt/ttsim-src-head
COPY soc_descriptor_bh.yaml /opt/sim/bh_head/soc_descriptor.yaml

# The 2-chip Blackhole (P300) simulator, for the mesh kernel -- prebuilt
# release binary (no source build needed for this one).
RUN mkdir -p /opt/sim/bh_x2 \
    && curl -sL -o /opt/sim/bh_x2/libttsim_bh_x2.so \
        https://github.com/tenstorrent/ttsim/releases/download/v1.10.1/libttsim_bh_x2.so
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
