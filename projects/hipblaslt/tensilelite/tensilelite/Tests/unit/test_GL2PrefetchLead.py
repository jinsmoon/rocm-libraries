# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the GL2 prefetch lead (PrefetchLeadCycles).

GL2PrefetchLoad.leadCycles turns the prefetch chunks in front of one tensor_load into the
cycles they need ahead of it; stinkytofu turns the cycles into WMMA windows. No GPU required.
"""

import os
import sys
from math import ceil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from tensilelite.Components.GL2Prefetch import GL2PrefetchLoad


def _kernel(numThreads=128, clusterDim=(1, 1)):
    return {"NumThreads": numThreads, "ClusterDim": list(clusterDim)}


def _tp(nc, nl=1):
    return {"gl2nc": nc, "gl2nl": nl}


def _cycles(chunks):
    """The model written out: one cycle per chunk plus the 75-cycle margin."""
    return chunks + 75


def test_chunks_per_inst_cluster_share():
    # 512 chunks over a 2x2 cluster of 128 threads: each workgroup takes 128.
    assert GL2PrefetchLoad.chunksPerInst(_kernel(clusterDim=(2, 2)), _tp(512)) == [128]
    # 64 chunks over 512 threads: 8 lanes share a chunk, 16 per workgroup.
    assert GL2PrefetchLoad.chunksPerInst(_kernel(clusterDim=(2, 2)), _tp(64)) == [16]
    # Chunks split evenly over the instructions; one workgroup of 128 threads covers each.
    assert GL2PrefetchLoad.chunksPerInst(_kernel(), _tp(300, nl=3)) == [100, 100, 100]
    assert GL2PrefetchLoad.chunksPerInst(_kernel(), _tp(250, nl=2)) == [125, 125]


def test_lead_cycles_whole_group_binds_when_largest_first():
    # A, B, MXSA, MXSB of an nvf4 2x2-cluster tile: 128 + 128 + 16 + 16 chunks.
    k = _kernel(clusterDim=(2, 2))
    tps = [_tp(512), _tp(512), _tp(64), _tp(64)]
    assert GL2PrefetchLoad.leadCycles(k, tps) == _cycles(288)


def test_lead_cycles_small_group():
    # 16 + 8 chunks: the whole group binds (24 > 8 * 2 / 1).
    k = _kernel()
    assert GL2PrefetchLoad.leadCycles(k, [_tp(16), _tp(8)]) == _cycles(24)


def test_lead_cycles_small_prefetches_first():
    # Scales issued first (splat order): the last two prefetches carry most chunks and get
    # half the lead, so their suffix binds instead of the whole group.
    k = _kernel(clusterDim=(2, 2))
    tps = [_tp(64), _tp(64), _tp(512), _tp(512)]
    suffix = 256 * 4 / 2
    assert GL2PrefetchLoad.leadCycles(k, tps) == ceil(suffix + 75)
    assert GL2PrefetchLoad.leadCycles(k, tps) > _cycles(288)
