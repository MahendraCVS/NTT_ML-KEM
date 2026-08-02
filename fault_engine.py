"""
NTTVis - Fault Injection Engine
===============================

This module implements fault injection simulations on the NTT/ML-KEM execution DAG.
Specifically, it implements the twiddle-pointer zeroization attack described in
"Fiddling the Twiddle Constants" (Ravi et al., 2022).
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any

import networkx as nx

from ntt_engine import NTTResult, ButterflyStep, PerformanceMetrics


def compute_entropy_profile(ntt_result: NTTResult) -> dict[int, float]:
    """Compute the normalized Shannon entropy of stage outputs stage-by-stage.

    Per stage, collects all values output by nodes in that stage, computes the
    Shannon entropy over the values multiset, and normalizes it.
    """
    values_by_stage: Dict[int, List[int]] = {}
    for step in ntt_result.execution_log:
        stage = step.stage_number
        if stage not in values_by_stage:
            values_by_stage[stage] = []
        values_by_stage[stage].extend(step.value_out)

    profile: Dict[int, float] = {}
    for stage, values in sorted(values_by_stage.items()):
        if not values:
            profile[stage] = 0.0
            continue

        counts = Counter(values)
        total = len(values)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Normalize Shannon entropy
        max_entropy = math.log2(total) if total > 1 else 1.0
        profile[stage] = entropy / max_entropy if max_entropy > 0 else 0.0

    return profile


class FaultInjector:
    """Simulates twiddle-pointer zeroization faults on the NTT execution DAG."""

    @staticmethod
    def _repropagate_graph(graph: nx.DiGraph, bit_reversed_input: List[int], modulus: int, n: int) -> List[int]:
        """Traverses the DAG in topological order and recomputes the node outputs

        using the current (possibly mutated) twiddle values. Updates the node
        attributes in-place.
        """
        memory = list(bit_reversed_input)
        topo_order = list(nx.topological_sort(graph))

        for node_id in topo_order:
            data = graph.nodes[node_id]
            op_type = data.get("op_type")
            inputs = data.get("inputs", [])
            outputs = data.get("outputs", [])
            twiddle = data.get("twiddle_value", 0)

            if not inputs or not outputs:
                continue

            if op_type in ("CT_DIT_BUTTERFLY", "GS_DIF_BUTTERFLY"):
                idx_a, idx_b = inputs[0], inputs[1]
                val_a_before = memory[idx_a]
                val_b_before = memory[idx_b]

                if op_type == "CT_DIT_BUTTERFLY":
                    t = (twiddle * val_b_before) % modulus
                    val_a_after = (val_a_before + t) % modulus
                    val_b_after = (val_a_before - t) % modulus
                else:  # GS_DIF_BUTTERFLY
                    val_a_after = (val_a_before + val_b_before) % modulus
                    val_b_after = (twiddle * (val_b_before - val_a_before)) % modulus

                memory[idx_a] = val_a_after
                memory[idx_b] = val_b_after

                data["value_in"] = (val_a_before, val_b_before)
                data["value_out"] = (val_a_after, val_b_after)

            elif op_type in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA"):
                idx0, idx1 = outputs[0], outputs[1]
                if len(inputs) == 4:
                    idx_a0, idx_a1, idx_b0, idx_b1 = inputs
                    a0, a1 = memory[idx_a0], memory[idx_a1]
                    b0, b1 = memory[idx_b0], memory[idx_b1]
                else:
                    idx_a0, idx_a1 = inputs
                    a0, a1 = memory[idx_a0], memory[idx_a1]
                    b0, b1 = a0, a1

                r0 = (a0 * b0 + twiddle * (a1 * b1)) % modulus
                r1 = (a0 * b1 + a1 * b0) % modulus

                memory[idx0] = r0
                memory[idx1] = r1

                data["value_in"] = (a0, a1)
                data["value_out"] = (r0, r1)

        # Apply scaling if GS inverse NTT is present in the pipeline
        has_gs = any(graph.nodes[nd].get("op_type") == "GS_DIF_BUTTERFLY" for nd in topo_order)
        if has_gs:
            inv_n_2 = pow(n // 2, modulus - 2, modulus)
            for i in range(n):
                memory[i] = (memory[i] * inv_n_2) % modulus

        return memory[:n]

    @staticmethod
    def _sync_execution_log(graph: nx.DiGraph, execution_log: List[ButterflyStep]) -> None:
        """Synchronizes execution log step attributes with the updated graph node values."""
        for step in execution_log:
            node_id = step.node_id
            if node_id in graph:
                data = graph.nodes[node_id]
                step.twiddle_value = data["twiddle_value"]
                step.value_in = data["value_in"]
                step.value_out = data["value_out"]

    @classmethod
    def zeroize_stage(cls, ntt_result: NTTResult, stage_number: int) -> NTTResult:
        """Sets twiddle_value = 0 for every node at the given stage_number

        and propagates the results forward through the DAG.
        """
        # Create deep copies to avoid mutating cached result
        new_graph = copy.deepcopy(ntt_result.graph)
        new_log = copy.deepcopy(ntt_result.execution_log)

        # Mutate twiddle values at the selected stage
        for node_id, data in new_graph.nodes(data=True):
            if data.get("stage_number") == stage_number:
                data["twiddle_value"] = 0

        # Repropagate values through mutated graph
        new_output = cls._repropagate_graph(
            new_graph,
            ntt_result.bit_reversed_input,
            ntt_result.modulus,
            ntt_result.n
        )

        # Update execution log
        cls._sync_execution_log(new_graph, new_log)

        return NTTResult(
            output=new_output,
            graph=new_graph,
            execution_log=new_log,
            bit_reversed_input=ntt_result.bit_reversed_input,
            n=ntt_result.n,
            modulus=ntt_result.modulus,
            root_of_unity=ntt_result.root_of_unity,
            performance_metrics=ntt_result.performance_metrics
        )

    @classmethod
    def zeroize_pointer(cls, ntt_result: NTTResult, stage_number: int, module_index: Optional[int] = None) -> NTTResult:
        """Models twiddle-pointer zeroization in a specific module.

        If module_index is None, defaults to 0.
        NTTs are partitioned into modules:
        - Module 0: NTT of A ("A_") and NTT of B ("B_")
        - Module 1: Inverse NTT of C ("B" GS_DIF_BUTTERFLY)
        """
        if module_index is None:
            module_index = 0

        new_graph = copy.deepcopy(ntt_result.graph)
        new_log = copy.deepcopy(ntt_result.execution_log)

        # Identify which nodes belong to the target module
        target_nodes = set()
        for node_id, data in new_graph.nodes(data=True):
            op_type = data.get("op_type")
            if op_type not in ("CT_DIT_BUTTERFLY", "GS_DIF_BUTTERFLY"):
                continue

            node_mod = 0
            if "A_" in node_id or "B_" in node_id:
                node_mod = 0
            elif "B" in node_id and op_type == "GS_DIF_BUTTERFLY":
                node_mod = 1
            else:
                node_mod = 0  # Standard NTT only has one module

            if node_mod == module_index:
                target_nodes.add(node_id)

        # Zeroize targeted nodes at the given stage_number
        for node_id in target_nodes:
            data = new_graph.nodes[node_id]
            if data.get("stage_number") == stage_number:
                data["twiddle_value"] = 0

        new_output = cls._repropagate_graph(
            new_graph,
            ntt_result.bit_reversed_input,
            ntt_result.modulus,
            ntt_result.n
        )

        cls._sync_execution_log(new_graph, new_log)

        return NTTResult(
            output=new_output,
            graph=new_graph,
            execution_log=new_log,
            bit_reversed_input=ntt_result.bit_reversed_input,
            n=ntt_result.n,
            modulus=ntt_result.modulus,
            root_of_unity=ntt_result.root_of_unity,
            performance_metrics=ntt_result.performance_metrics
        )
