# ttsim-blackhole -- real ttnn code executed against a simulated Tenstorrent
# Blackhole chip, no Tenstorrent hardware anywhere in this container.
#
# Base image already ships a self-contained, pip-installed ttnn wheel (bundles
# its own kernel/runtime source), so no TT_METAL_HOME checkout or from-source
# tt-metal build is needed. All runtime shared libraries this stack needs
# (libmpi, libhwloc, libnuma, libevent-core/pthreads) are already present.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

# Build ttsim's Blackhole simulator library from source, pinned to v1.10.1.
# NOT the latest tag: v1.10.5 (current HEAD as of this writing) has a
# reproduced regression on Blackhole -- any ttnn.matmul with an output free
# dimension of 3072 (e.g. a GPT-2-style MLP up-projection, 768 -> 3072)
# aborts with `UnsupportedFunctionality: tensix_pacr: Disable_pack_zero_flags`,
# even though the same op sequence at narrower widths (768) is fine. Re-test
# against a newer tag before moving this pin.
RUN git clone --depth=1 --branch v1.10.1 https://github.com/tenstorrent/ttsim.git /opt/ttsim-src \
    && cd /opt/ttsim-src \
    && ./make.py src/_out/release_bh/libttsim.so \
    && mkdir -p /opt/sim/bh \
    && cp src/_out/release_bh/libttsim.so /opt/sim/bh/libttsim_bh.so \
    && rm -rf /opt/ttsim-src

# ttsim resolves the SOC descriptor as a sibling of the .so file, named
# exactly `soc_descriptor.yaml`.
COPY soc_descriptor_bh.yaml /opt/sim/bh/soc_descriptor.yaml

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
