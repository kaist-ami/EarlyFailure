# ViCLIP attribution

The Python sources in this directory are minimally adapted from
[`Vchitect/VBench`](https://github.com/Vchitect/VBench), commit
`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`, directory
`vbench/third_party/ViCLIP`.

VBench is distributed under the Apache License 2.0. A copy is included as
`LICENSE`. The adaptations remove VBench-wide imports/download subprocesses and
load inference weights with PyTorch's `weights_only=True`; model architecture
and parameter names are unchanged.

