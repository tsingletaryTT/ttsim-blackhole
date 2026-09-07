---
title: ttsim-blackhole
emoji: 🕳️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ttsim-blackhole

**Live demo:** https://huggingface.co/spaces/episod/ttsim-blackhole

Real [ttnn](https://github.com/tenstorrent/tt-metal) code, executed against a
**simulated Tenstorrent Blackhole chip** via
[ttsim](https://github.com/tenstorrent/ttsim) — no Tenstorrent hardware
anywhere in this container. Type code into the editor, pick a backend, hit
Run — it streams back real output from a real (virtual) chip.

Backed by Tenstorrent's official pip-installable `ttnn` wheel plus ttsim
built from source (pinned to `v1.10.1` — a later tag has a reproduced
Blackhole regression on wide matmuls, see the Dockerfile).
