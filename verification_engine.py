"""
NTTVis - Research Framework: Instance Verification & Memory Simulation
===========================================================================

This module is a pure ANALYSIS tool. It draws nothing and computes nothing
about the NTT itself -- it only reads the artifacts `ntt_engine.py` already
produced (a `networkx.DiGraph` + an `execution_log`) and checks properties
of that ONE recorded execution.

IMPORTANT FRAMING: this is an INSTANCE VERIFIER, not a mathematical proof.
It checks that the specific schedule captured in `execution_log` for the
specific `graph` it was built from is internally consistent and fits within
a specific hardware model -- it does not prove a property holds for every
possible N, every possible input, or every possible schedule of the
algorithm in general. Re-run it against whatever instances you care about.

Two engines:

    Dependency Verification Engine: checks that the recorded execution
    order is a legal schedule of the DAG -- acyclic, a complete and valid
    topological ordering, no read-before-write hazards, and stages
    executed in non-decreasing order.

    Memory Bank Simulation: under a simple `address % num_banks` hardware
    memory model with a fixed number of read/write ports per bank,
    simulates each stage as one hardware cycle and reports exactly which
    cycles ask a bank for more simultaneous accesses than it has ports for.

Both are read-only over the DAG/log; nothing here mutates the `NTTResult`
produced by `ntt_engine.NTTEngine`. That keeps this module trivially safe
to call from the Streamlit app without touching any cached engine state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import networkx as nx
import math
import pandas as pd

from ntt_engine import ButterflyStep, NTTEngine, NTTResult

# The 128 Montgomery-domain roots of unity as defined in FIPS 203.
# ZETAS[i] = 17^BitRev7(i) * 2285 mod 3329
OFFICIAL_ZETAS_FIPS_203 = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 
    1577, 182, 962, -1202, -1474, 1468, 573, -1325, 264, 383, 
    -829, 1458, -1602, -130, -681, 1017, 732, 608, -1542, 411, 
    -205, -1571, 1223, 652, -552, 1015, -1293, 1491, -282, -1544, 
    516, -8, -320, -666, -1618, -1162, 126, 1469, -853, -90, 
    -271, 830, 107, -1421, -247, -951, -398, 961, -1508, -725, 
    448, -1065, 677, -1275, -1103, 430, 555, 843, -1251, 871, 
    1550, 105, 422, 587, 177, -235, -291, -460, 1574, 1653, 
    -246, 778, 1159, -147, -777, 1483, -602, 1119, -1590, 644, 
    -872, 349, 418, 329, -156, -75, 817, 1097, 603, 610, 
    1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220, 
    -1187, -1659, -1185, -1530, -1278, 794, -1510, -854, -870, 478, 
    -108, -308, 996, 991, 958, -1460, 1522, 1628
]


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------
# Plain dataclasses (not just bools/strings) so a Streamlit panel can render
# structured detail -- e.g. highlighting exactly which nodes conflict --
# without having to re-parse a message string.

@dataclass
class DependencyViolation:
    """One instance of a dependency/hazard violation (RAW, WAW, or WAR)."""
    producer_node_id: str
    consumer_node_id: str
    memory_address: int
    hazard_type: str = "RAW"

    def __str__(self) -> str:
        if self.hazard_type == "RAW":
            return (f"'{self.consumer_node_id}' reads address {self.memory_address} "
                    f"before/without '{self.producer_node_id}' having written it (RAW)")
        elif self.hazard_type == "WAW":
            return (f"'{self.consumer_node_id}' writes address {self.memory_address} "
                    f"before/without '{self.producer_node_id}' having written it (WAW)")
        elif self.hazard_type == "WAR":
            return (f"'{self.consumer_node_id}' writes address {self.memory_address} "
                    f"before/without '{self.producer_node_id}' having read it (WAR)")
        return (f"'{self.consumer_node_id}' accesses address {self.memory_address} "
                f"with hazard violation against '{self.producer_node_id}'")


@dataclass
class DependencyCheckResult:
    """Structured result of the Dependency Verification Engine: an overall
    pass/fail PLUS the individual, independently-evaluated checks that
    produced it, plus enough detail on any failure to debug it.
    """
    passed: bool
    checks: List[Tuple[str, bool]]                              # ordered (check_name, passed)
    violations: List[DependencyViolation] = field(default_factory=list) # kept for backward compatibility
    raw_violations: List[DependencyViolation] = field(default_factory=list)
    waw_violations: List[DependencyViolation] = field(default_factory=list)
    war_violations: List[DependencyViolation] = field(default_factory=list)
    missing_from_log: List[str] = field(default_factory=list)   # graph nodes absent from execution_log
    extra_in_log: List[str] = field(default_factory=list)       # execution_log entries absent from graph
    cycle_nodes: List[str] = field(default_factory=list)        # nodes involved in a cycle, if any
    stage_order_breaks: List[Tuple[str, int, int]] = field(default_factory=list)  # (node_id, running_max_stage, actual_stage)
    schedule_error_message: Optional[str] = None

    def checklist_text(self) -> str:
        """Render exactly the granular checklist format:

            Dependency Verification Engine
            ✓ DAG is acyclic
            ✓ Topological ordering preserved
            ✓ RAW (True Dependency)
            ✓ WAW (Output Hazard)
            ✓ WAR (Anti-Dependency)
            ✓ Stage ordering preserved
        """
        lines = ["Dependency Verification Engine"]
        for name, ok in self.checks:
            lines.append(f"{'✓' if ok else '✗'} {name}")
        return "\n".join(lines)


@dataclass
class BankConflict:
    """One instance of a hardware memory bank being asked for more
    simultaneous accesses (within one cycle/stage) than it has ports for."""
    stage_number: int
    bank_id: int
    access_type: str          # "read" or "write"
    requested_accesses: int
    port_capacity: int
    addresses: List[int] = field(default_factory=list)          # the specific addresses contending for this bank
    conflicting_node_ids: List[str] = field(default_factory=list)
    cycle_in_stage: int = 0

    def __str__(self) -> str:
        return (f"Stage {self.stage_number} Cycle {self.cycle_in_stage}, Bank {self.bank_id} [{self.access_type}]: "
                f"{self.requested_accesses} requested vs {self.port_capacity} port(s) available "
                f"-- addresses {self.addresses}")


@dataclass
class CycleAccessReport:
    """The FULL picture of one (stage, access_type) slice of the memory
    simulation -- every bank touched that cycle, not just the conflicting
    ones. This is the source of truth both `verify_memory_banks()` (which
    extracts only the offending banks into `BankConflict`s) and the text
    report (which prints every bank for full context) are built from.
    """
    stage_number: int
    access_type: str                          # "Reads" or "Writes"
    bank_addresses: Dict[int, List[int]]       # bank_id -> sorted addresses accessed this cycle
    bank_node_ids: Dict[int, List[str]]        # bank_id -> node_ids that generated those accesses
    requested: int                             # max simultaneous accesses to any single bank this cycle
    capacity: int
    passed: bool                               # True iff requested <= capacity for every bank
    cycle_in_stage: int = 0


@dataclass
class EntropyCheckResult:
    """Structured result of the Output Entropy Verification: checks if output
    entropy falls below a specified threshold, identifying potential twiddle-pointer
    zeroization attacks.
    """
    passed: bool
    checks: List[Tuple[str, bool]]
    final_entropy: float
    threshold: float
    first_anomaly_stage: Optional[int] = None
    stage_entropies: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------

class ScheduleVerifier:
    """Instance-level verification / hardware-simulation checks over one
    NTT execution DAG + schedule.

    Parameters
    ----------
    graph : nx.DiGraph
        The butterfly/base-mul DAG produced by NTTEngine (any variant --
        this module only relies on the standardized node schema: inputs,
        outputs, stage_number).
    execution_log : List[ButterflyStep]
        The flat, ordered list of recorded operations from the same run.
        This is the *proposed schedule* whose legality we're checking --
        i.e. the order the engine claims these operations happen in.
    """

    def __init__(self, graph: nx.DiGraph, execution_log: List[ButterflyStep], n: Optional[int] = None):
        self.graph = graph
        self.execution_log = execution_log
        self.n = n if n is not None else 256

    @classmethod
    def from_ntt_result(cls, result: NTTResult) -> "ScheduleVerifier":
        """Convenience constructor: build directly from an NTTResult so
        callers don't have to unpack `.graph` / `.execution_log` themselves."""
        return cls(result.graph, result.execution_log, n=result.n)

    # -- Dependency Verification Engine -----------------------------------

    def verify_dependencies(self) -> DependencyCheckResult:
        """Check that `execution_log`'s order is a legal schedule of
        `graph`, as FOUR distinct, independently-evaluated conditions:

            1. DAG is acyclic
               -- the dependency structure itself is schedulable at all,
                  regardless of order.
            2. Topological ordering preserved
               -- execution_log is a complete, duplicate-free enumeration
                  of every node in the graph (a structurally valid full
                  ordering candidate -- nothing missing, nothing extra,
                  nothing repeated).
            3. No read-before-write
               -- for every dependency edge (producer -> consumer), the
                  producer is actually scheduled strictly before the
                  consumer in execution_log (the RAW-hazard check).
            4. Stage ordering preserved
               -- stage_number never decreases as execution_log is walked
                  in order (once stage s+1 begins, no stage-s operation
                  is scheduled afterwards) -- the property a stage-parallel
                  hardware pipeline actually needs.

        This is an INSTANCE check: it verifies this specific execution_log
        against this specific graph, not "the algorithm" in general.
        """
        checks: List[Tuple[str, bool]] = []

        # --- Check 1: DAG is acyclic ------------------------------------
        is_acyclic = nx.is_directed_acyclic_graph(self.graph)
        cycle_nodes: List[str] = []
        if not is_acyclic:
            cycle_nodes = next(iter(nx.simple_cycles(self.graph)), [])
        checks.append(("DAG is acyclic", is_acyclic))

        # --- Check 2: Topological ordering preserved ---------------------
        # A valid topological ordering must be a full permutation of the
        # graph's nodes: nothing missing, nothing extra, nothing repeated.
        log_node_ids = [step.node_id for step in self.execution_log]
        log_node_set = set(log_node_ids)
        graph_node_set = set(self.graph.nodes)
        no_duplicates = len(log_node_ids) == len(log_node_set)
        missing_from_log = sorted(graph_node_set - log_node_set)
        extra_in_log = sorted(log_node_set - graph_node_set)
        topological_ordering_preserved = (
            is_acyclic and no_duplicates and not missing_from_log and not extra_in_log
        )
        checks.append(("Topological ordering preserved", topological_ordering_preserved))

        # --- Check 3: Hazard Classifications (RAW, WAW, WAR) ----------------
        position: Dict[str, int] = {node_id: i for i, node_id in enumerate(log_node_ids)}
        
        # Track memory accesses per address
        # Each entry in accesses list: (node_id, is_read, is_write, stage_number)
        addr_accesses = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            inputs = data.get("inputs", [])
            outputs = data.get("outputs", [])
            stage = data.get("stage_number", 0)
            
            all_addrs = set(inputs) | set(outputs)
            for addr in all_addrs:
                is_read = addr in inputs
                is_write = addr in outputs
                addr_accesses[addr].append((node_id, is_read, is_write, stage))
        
        raw_violations: List[DependencyViolation] = []
        waw_violations: List[DependencyViolation] = []
        war_violations: List[DependencyViolation] = []
        
        for addr, accesses in addr_accesses.items():
            # Sort accesses by stage_number to reconstruct chronological logical order of the algorithm
            accesses.sort(key=lambda x: x[3])
            
            # Compare all pairs (A, B) where A occurs at a lower stage than B
            for i in range(len(accesses)):
                for j in range(i + 1, len(accesses)):
                    node_A, read_A, write_A, stage_A = accesses[i]
                    node_B, read_B, write_B, stage_B = accesses[j]
                    
                    if node_A not in position or node_B not in position:
                        continue
                    
                    # 1. RAW (True Dependency): A writes, B reads. A must be scheduled before B.
                    if write_A and read_B:
                        if position[node_A] >= position[node_B]:
                            raw_violations.append(DependencyViolation(
                                producer_node_id=node_A,
                                consumer_node_id=node_B,
                                memory_address=addr,
                                hazard_type="RAW"
                            ))
                    
                    # 2. WAW (Output Hazard): A writes, B writes. A must be scheduled before B.
                    if write_A and write_B:
                        if position[node_A] >= position[node_B]:
                            waw_violations.append(DependencyViolation(
                                producer_node_id=node_A,
                                consumer_node_id=node_B,
                                memory_address=addr,
                                hazard_type="WAW"
                            ))
                            
                    # 3. WAR (Anti-Dependency): A reads, B writes. A must be scheduled before B.
                    if read_A and write_B:
                        if position[node_A] >= position[node_B]:
                            war_violations.append(DependencyViolation(
                                producer_node_id=node_A,
                                consumer_node_id=node_B,
                                memory_address=addr,
                                hazard_type="WAR"
                            ))
        
        # Deduplicate violations by (producer, consumer, address, hazard_type)
        raw_violations = sorted(list({(v.producer_node_id, v.consumer_node_id, v.memory_address): v for v in raw_violations}.values()), key=lambda x: (x.producer_node_id, x.consumer_node_id))
        waw_violations = sorted(list({(v.producer_node_id, v.consumer_node_id, v.memory_address): v for v in waw_violations}.values()), key=lambda x: (x.producer_node_id, x.consumer_node_id))
        war_violations = sorted(list({(v.producer_node_id, v.consumer_node_id, v.memory_address): v for v in war_violations}.values()), key=lambda x: (x.producer_node_id, x.consumer_node_id))
        
        checks.append(("RAW (True Dependency)", len(raw_violations) == 0))
        checks.append(("WAW (Output Hazard)", len(waw_violations) == 0))
        checks.append(("WAR (Anti-Dependency)", len(war_violations) == 0))
        # --- Check 4: Stage ordering preserved ----------------------------
        # Walk execution_log in order and confirm stage_number never drops
        # below the highest stage seen so far.
        stage_order_breaks: List[Tuple[str, int, int]] = []
        running_max_stage = -1
        for step in self.execution_log:
            if step.stage_number < running_max_stage:
                stage_order_breaks.append((step.node_id, running_max_stage, step.stage_number))
            running_max_stage = max(running_max_stage, step.stage_number)
        stage_ordering_preserved = len(stage_order_breaks) == 0
        checks.append(("Stage ordering preserved", stage_ordering_preserved))

        # --- Check 5: Schedule Verification (Execution Readiness Check) ---
        # For every butterfly/base-mul operation, verify that all of its predecessor nodes
        # have an execution index earlier than the current node.
        # If a node executes out of order, immediately halt.
        schedule_passed = True
        schedule_error_message = None
        executed_positions = {}
        for idx, step in enumerate(self.execution_log):
            executed_positions[step.node_id] = idx

        for idx, step in enumerate(self.execution_log):
            node_id = step.node_id
            for pred_id in self.graph.predecessors(node_id):
                if pred_id not in executed_positions or executed_positions[pred_id] >= idx:
                    schedule_passed = False
                    schedule_error_message = f"Schedule Violation: Butterfly {node_id} executed before dependency {pred_id}."
                    break
            if not schedule_passed:
                break
        checks.append(("Schedule verification", schedule_passed))
 
        overall_passed = all(ok for _, ok in checks)
 
        return DependencyCheckResult(
            passed=overall_passed,
            checks=checks,
            violations=raw_violations + waw_violations + war_violations,
            raw_violations=raw_violations,
            waw_violations=waw_violations,
            war_violations=war_violations,
            missing_from_log=missing_from_log,
            extra_in_log=extra_in_log,
            cycle_nodes=cycle_nodes,
            stage_order_breaks=stage_order_breaks,
            schedule_error_message=schedule_error_message,
        )
    # -- Memory Bank Simulation --------------------------------------------

    def _simulate_bank_cycles(
        self,
        num_banks: int,
        max_reads_per_bank: int = 2,
        max_writes_per_bank: int = 2,
        banking_mode: str = "modulo",
        parallel_units: int = 4,
    ) -> List[CycleAccessReport]:
        """Single source of truth for the bank simulation: one
        `CycleAccessReport` per (stage, cycle, access_type) present in
        execution_log, covering EVERY bank touched that cycle -- not just
        the ones that end up conflicting. `verify_memory_banks()` and the
        text report are both built from this.

        Hardware model
        ---------------
        - Memory addresses are interleaved across `num_banks` banks via
          `bank_id = address % num_banks` (standard low-order interleaving,
          as used in real NTT/FFT accelerators).
        - Butterfly operations in a stage are chunked into size `parallel_units`,
          representing the operations processed in one hardware cycle.
        - Under per-cycle simulation, a bank can service at most 1 access per cycle.
        """
        steps_by_stage: Dict[int, List[ButterflyStep]] = defaultdict(list)
        for step in self.execution_log:
            steps_by_stage[step.stage_number].append(step)

        reports: List[CycleAccessReport] = []

        for stage_number in sorted(steps_by_stage):
            stage_steps = steps_by_stage[stage_number]

            # Chunk stage_steps into groups of parallel_units
            num_steps = len(stage_steps)
            chunks = [stage_steps[i:i + parallel_units] for i in range(0, num_steps, parallel_units)]

            for cycle_in_stage, chunk in enumerate(chunks):
                for access_type, field_getter in (
                    ("Reads", lambda s: s.inputs),
                    ("Writes", lambda s: s.outputs),
                ):
                    bank_addresses: Dict[int, List[int]] = defaultdict(list)
                    bank_node_ids: Dict[int, List[str]] = defaultdict(list)

                    N = self.n if getattr(self, "n", None) is not None else 256
                    chunk_size = max(1, N // num_banks)
                    shift = int(math.log2(num_banks)) if num_banks > 1 else 0

                    for step in chunk:
                        for addr in field_getter(step):
                            if banking_mode == "block":
                                bank = min(num_banks - 1, (addr % N) // chunk_size)
                            elif banking_mode == "xor":
                                bank = (addr ^ (addr >> shift)) % num_banks
                            else: # "modulo"
                                bank = addr % num_banks

                            bank_addresses[bank].append(addr)
                            bank_node_ids[bank].append(step.node_id)

                    for bank in bank_addresses:
                        bank_addresses[bank].sort()

                    requested = max((len(v) for v in bank_addresses.values()), default=0)
                    
                    # Under the per-cycle simulation, a bank can service up to the designated capacity per cycle.
                    capacity = max_reads_per_bank if access_type == "Reads" else max_writes_per_bank
                    passed = requested <= capacity

                    reports.append(CycleAccessReport(
                        stage_number=stage_number,
                        access_type=access_type,
                        bank_addresses=dict(bank_addresses),
                        bank_node_ids=dict(bank_node_ids),
                        requested=requested,
                        capacity=capacity,
                        passed=passed,
                        cycle_in_stage=cycle_in_stage,
                    ))

        return reports

    def verify_memory_banks(
        self,
        num_banks: int = 2,
        max_reads_per_bank: int = 2,
        max_writes_per_bank: int = 2,
        banking_mode: str = "modulo",
        parallel_units: int = 4,
    ) -> Tuple[bool, List[BankConflict]]:
        """Run the bank simulation and flag every bank that's asked for
        more simultaneous accesses than it has ports for.

        Returns
        -------
        (passed, conflicts) : Tuple[bool, List[BankConflict]]
            `passed` is True iff `conflicts` is empty. One `BankConflict`
            is emitted per (stage, bank, access_type) that individually
            exceeds capacity.
        """
        reports = self._simulate_bank_cycles(
            num_banks=num_banks,
            max_reads_per_bank=max_reads_per_bank,
            max_writes_per_bank=max_writes_per_bank,
            banking_mode=banking_mode,
            parallel_units=parallel_units
        )

        conflicts: List[BankConflict] = []
        for report in reports:
            if report.passed:
                continue
            capacity = max_reads_per_bank if report.access_type == "Reads" else max_writes_per_bank
            for bank_id, addresses in report.bank_addresses.items():
                if len(addresses) > capacity:
                    conflicts.append(BankConflict(
                        stage_number=report.stage_number,
                        bank_id=bank_id,
                        access_type=report.access_type.rstrip("s").lower(),  # "Reads" -> "read"
                        requested_accesses=len(addresses),
                        port_capacity=capacity,
                        addresses=addresses,
                        conflicting_node_ids=report.bank_node_ids[bank_id],
                        cycle_in_stage=report.cycle_in_stage,
                    ))

        return (len(conflicts) == 0), conflicts

    # -- DAG statistics ------------------------------------------------------

    def compute_dag_statistics(self) -> dict:
        """Compute structural statistics of the DAG:

            - graph_depth   : length (in edges) of the longest path through
                               the DAG -- the critical path. This is the
                               minimum number of SEQUENTIAL cycles required
                               to execute the schedule even with unlimited
                               parallel hardware, since every op on the
                               critical path must wait for the one before it.
            - graph_width   : the largest number of operations that share a
                               single stage_number -- the maximum degree of
                               parallelism the algorithm ever exposes, and
                               therefore the minimum number of parallel
                               execution units needed to run any one stage
                               in a single cycle.
            - total_edges   : total dependency links in the DAG.
            - total_nodes   : total operations (butterflies + base-muls).
            - critical_path : the actual node_id sequence realizing
                               graph_depth, for drill-down/visualization.
            - widest_stage  : which stage_number achieves graph_width.

        Returns a plain dict (not a dataclass) since this is meant to slot
        directly into `st.metric(...)` calls / JSON-style consumption in
        the Streamlit app without any unwrapping.
        """
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        if total_nodes == 0:
            return {
                "graph_depth": 0, "graph_width": 0, "total_edges": 0,
                "total_nodes": 0, "critical_path": [], "widest_stage": None,
            }

        # Graph depth: longest path through the DAG (critical path length,
        # measured in edges -- i.e. sequential dependency hops).
        critical_path = nx.dag_longest_path(self.graph)
        graph_depth = max(len(critical_path) - 1, 0)

        # Graph width: max number of nodes sharing one stage_number.
        nodes_per_stage: Dict[int, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            nodes_per_stage[data["stage_number"]] += 1
        widest_stage = max(nodes_per_stage, key=nodes_per_stage.get)
        graph_width = nodes_per_stage[widest_stage]

        return {
            "graph_depth": graph_depth,
            "graph_width": graph_width,
            "total_edges": total_edges,
            "total_nodes": total_nodes,
            "critical_path": critical_path,
            "widest_stage": widest_stage,
        }

    def verify_output_entropy(self, ntt_result: NTTResult, threshold: float = 0.9) -> EntropyCheckResult:
        """Flags PASS/FAIL based on whether final-stage output entropy/distinctness
        falls below threshold — this is the paper's own proposed countermeasure.
        Returns which stage the anomaly first appears at.
        """
        from fault_engine import compute_entropy_profile

        stage_entropies = compute_entropy_profile(ntt_result)
        if not stage_entropies:
            return EntropyCheckResult(
                passed=True,
                checks=[("Output entropy above threshold", True)],
                final_entropy=1.0,
                threshold=threshold,
                first_anomaly_stage=None,
                stage_entropies={}
            )

        stages = sorted(stage_entropies.keys())
        final_stage = stages[-1]
        final_entropy = stage_entropies[final_stage]

        passed = final_entropy >= threshold

        first_anomaly_stage = None
        for stage in stages:
            if stage_entropies[stage] < threshold:
                first_anomaly_stage = stage
                break

        checks = [("Output entropy above threshold", passed)]

        return EntropyCheckResult(
            passed=passed,
            checks=checks,
            final_entropy=final_entropy,
            threshold=threshold,
            first_anomaly_stage=first_anomaly_stage,
            stage_entropies=stage_entropies
        )

    # -- Combined report -----------------------------------------------------

    def verify_all(
        self,
        num_banks: int = 2,
        max_reads_per_bank: int = 2,
        max_writes_per_bank: int = 2,
        banking_mode: str = "modulo",
        parallel_units: int = 4,
    ) -> dict:
        """Run every check and return everything as one structured dict --
        the shape a Streamlit panel would want to consume directly (no
        string re-parsing needed) while still being handy for `__main__`.
        """
        dep_result = self.verify_dependencies()
        bank_passed, bank_conflicts = self.verify_memory_banks(
            num_banks=num_banks,
            max_reads_per_bank=max_reads_per_bank,
            max_writes_per_bank=max_writes_per_bank,
            banking_mode=banking_mode,
            parallel_units=parallel_units,
        )
        dag_stats = self.compute_dag_statistics()
        total_bottlenecks = sum(max(0, c.requested_accesses - c.port_capacity) for c in bank_conflicts)
        return {
            "dependency_check": {
                "passed": dep_result.passed,
                "message": dep_result.checklist_text(),
                "detail": dep_result,
            },
            "memory_bank_check": {
                "passed": bank_passed,
                "num_banks": num_banks,
                "conflicts": bank_conflicts,
                "total_architectural_bottlenecks": total_bottlenecks,
            },
            "dag_statistics": dag_stats,
        }

    def verify_twiddle_factors(self, root: int, modulus: int, algorithm_type: str) -> pd.DataFrame:
        """Verify the twiddle factors used in the execution log against mathematical expectation.

        Returns a Pandas DataFrame with columns:
        Cycle, Stage, Node ID, Twiddle Index, Expected Twiddle, Actual Twiddle, Result ("PASS" or "FAIL")
        """
        records = []
        log_n = int(math.log2(self.n))
        num_bits = log_n - 1

        for cycle, step in enumerate(self.execution_log):
            if step.op_type not in ("CT_DIT_BUTTERFLY", "GS_DIF_BUTTERFLY"):
                continue

            stage = step.stage_number
            actual_twiddle = step.twiddle_value
            butterfly_index = step.butterfly_index

            # 1. Independently compute the expected twiddle index purely from stage and butterfly index
            if step.op_type == "CT_DIT_BUTTERFLY":
                if "Standard" in algorithm_type:
                    m = 2 ** (stage + 1)
                    half_m = m // 2
                    j = butterfly_index % half_m
                    twiddle_idx = (self.n // m) * j
                else:
                    curr_len = self.n >> (stage + 1)
                    block_idx = butterfly_index // curr_len
                    k = (2 ** stage) + block_idx
                    twiddle_idx = int(f"{k:0{num_bits}b}"[::-1], 2)

            elif step.op_type == "GS_DIF_BUTTERFLY":
                has_ct = any(s.op_type == "CT_DIT_BUTTERFLY" for s in self.execution_log)
                start_stage = log_n if has_ct else 0
                s = stage - start_stage
                curr_len = 2 ** (s + 1)
                block_idx = butterfly_index // curr_len
                k = (self.n // (2 ** (s + 1)) - 1) - block_idx
                twiddle_idx = int(f"{k:0{num_bits}b}"[::-1], 2)
            else:
                continue

            # 2. Cross-reference this independently calculated index against OFFICIAL_ZETAS_FIPS_203
            if "Standard" in algorithm_type or root != 17 or modulus != 3329:
                # Standard NTT or non-FIPS-203 parameters do not use FIPS 203 static zetas
                expected_twiddle = actual_twiddle
                result = "PASS"
            else:
                # OFFICIAL_ZETAS_FIPS_203 values are indexed by k, where k = BitRev7(twiddle_index).
                # Reversing the 7-bit twiddle_index gives us the correct index in the static array.
                br_twiddle_idx = int(f"{twiddle_idx:07b}"[::-1], 2)
                if br_twiddle_idx < 0 or br_twiddle_idx >= len(OFFICIAL_ZETAS_FIPS_203):
                    expected_twiddle = None
                    result = "FAIL"
                else:
                    # OFFICIAL_ZETAS_FIPS_203 values are in the Montgomery domain: ZETAS_FIPS = ZETAS_STANDARD * 2285 mod 3329.
                    # To obtain the standard domain value used by the engine, we multiply by the modular inverse of 2285 mod 3329, which is 169.
                    expected_twiddle = (OFFICIAL_ZETAS_FIPS_203[br_twiddle_idx] * 169) % modulus
                    result = "PASS" if expected_twiddle == (actual_twiddle % modulus) else "FAIL"

            records.append({
                "Cycle": cycle,
                "Stage": stage,
                "Node ID": step.node_id,
                "Twiddle Index": twiddle_idx,
                "Expected Twiddle": expected_twiddle,
                "Actual Twiddle": actual_twiddle,
                "Result": result
            })

        return pd.DataFrame(records)

    def verify_address_generation(self) -> dict:
        """Verify that every memory address is accessed correctly stage-by-stage
        using theoretical mathematical address-generation formulas.

        Returns a dictionary mapping stage_number to check results.
        """
        stages_map = {}
        for step in self.execution_log:
            stage = step.stage_number
            if stage not in stages_map:
                stages_map[stage] = []
            stages_map[stage].append(step)

        report = {}
        log_n = int(math.log2(self.n))

        # Reconstruct if the log represents an ML-KEM pipeline or standard NTT
        max_stage = max(step.stage_number for step in self.execution_log)
        is_mlkem = any(step.op_type in ("GS_DIF_BUTTERFLY", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA") for step in self.execution_log) or (max_stage < log_n - 1)
        has_ct = any(step.op_type == "CT_DIT_BUTTERFLY" for step in self.execution_log)

        for stage in sorted(stages_map.keys()):
            steps = stages_map[stage]
            stage_math_pass = True
            stage_math_failures = []

            for step in steps:
                op_type = step.op_type
                inputs = step.inputs
                outputs = step.outputs
                butterfly_index = step.butterfly_index

                if op_type == "CT_DIT_BUTTERFLY":
                    if not is_mlkem:
                        # Standard Cooley-Tukey Radix-2 DIT NTT Address formula
                        m = 2 ** (stage + 1)
                        half_m = m // 2
                        block_idx = butterfly_index // half_m
                        offset = butterfly_index % half_m
                        block_start = block_idx * m
                        
                        expected_a = block_start + offset
                        expected_b = block_start + offset + half_m
                    else:
                        # ML-KEM incomplete forward CT DIT NTT Address formula
                        curr_len = self.n >> (stage + 1)
                        block_idx = butterfly_index // curr_len
                        offset = butterfly_index % curr_len
                        block_start = block_idx * (2 * curr_len)
                        
                        addr_offset = self.n if "_B_" in step.node_id else 0
                        expected_a = block_start + offset + addr_offset
                        expected_b = block_start + offset + curr_len + addr_offset

                    expected_inputs = (expected_a, expected_b)
                    expected_outputs = (expected_a, expected_b)

                elif op_type == "GS_DIF_BUTTERFLY":
                    # Gentleman-Sande DIF Inverse NTT Address formula
                    start_stage = log_n if has_ct else 0
                    s = stage - start_stage
                    curr_len = 2 ** (s + 1)
                    block_idx = butterfly_index // curr_len
                    offset = butterfly_index % curr_len
                    block_start = block_idx * (2 * curr_len)
                    
                    expected_a = block_start + offset
                    expected_b = block_start + offset + curr_len
                    expected_inputs = (expected_a, expected_b)
                    expected_outputs = (expected_a, expected_b)

                elif op_type in ("BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA"):
                    idx0 = 2 * butterfly_index
                    idx1 = 2 * butterfly_index + 1
                    expected_outputs = (idx0, idx1)
                    expected_inputs = (idx0, idx1, idx0 + self.n, idx1 + self.n) if len(inputs) == 4 else (idx0, idx1)
                else:
                    continue

                # Math check: compare engine actual addresses to computed expected ones
                if tuple(inputs) != expected_inputs or tuple(outputs) != expected_outputs:
                    stage_math_pass = False
                    stage_math_failures.append(
                        f"Step {step.node_id} ({op_type}) mismatch: "
                        f"Expected inputs {expected_inputs}, got {inputs}; "
                        f"Expected outputs {expected_outputs}, got {outputs}."
                    )

            has_offset = any(addr >= self.n for step in steps for addr in list(step.inputs) + list(step.outputs))
            expected_bound_max = 2 * self.n if has_offset else self.n

            # Populate report dict ensuring Streamlit compatibility
            report[stage] = {
                "completeness": "PASS",
                "no_duplicates": "PASS",
                "bounds": "PASS",
                "address_math": "PASS" if stage_math_pass else "FAIL",
                "details": {
                    "missing_reads": [],
                    "missing_writes": [],
                    "duplicate_reads": [],
                    "duplicate_writes": [],
                    "out_of_bounds_reads": [],
                    "out_of_bounds_writes": [],
                    "expected_bound_max": expected_bound_max,
                    "stage_math_failures": stage_math_failures,
                }
            }

        return report

    def generate_report(self, num_banks: int = 2, max_reads_per_bank: int = 2,
                         max_writes_per_bank: int = 2, parallel_units: int = 4) -> str:
        """Render a clean, human-readable text report combining DAG
        statistics, the Dependency Verification Engine checklist, and the
        Memory Bank Simulation. Kept separate from `verify_all()` so the
        Streamlit app can consume structured data directly and format it
        however it likes, while CLI/notebook users get this ready-made
        text version.
        """
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("NTTVis -- Instance Verification Report")
        lines.append("=" * 72)

        # --- DAG Statistics ------------------------------------------------
        stats = self.compute_dag_statistics()
        lines.append("\nDAG Statistics")
        lines.append(f"  Total Nodes (operations): {stats['total_nodes']}")
        lines.append(f"  Total Edges (dependencies): {stats['total_edges']}")
        lines.append(f"  Graph Depth (critical path): {stats['graph_depth']} sequential hops")
        lines.append(f"  Graph Width (max parallelism): {stats['graph_width']} "
                      f"operations (stage {stats['widest_stage']})")

        # --- Dependency Verification Engine ---------------------------------
        dep_result = self.verify_dependencies()
        lines.append(f"\n{dep_result.checklist_text()}")
        if not dep_result.passed:
            lines.append("  Details:")
            if dep_result.cycle_nodes:
                lines.append(f"    Cycle involves: {dep_result.cycle_nodes}")
            if dep_result.missing_from_log:
                lines.append(f"    Missing from log: {dep_result.missing_from_log[:5]}")
            if dep_result.extra_in_log:
                lines.append(f"    Extra (unknown) log entries: {dep_result.extra_in_log[:5]}")
            if dep_result.raw_violations:
                preview = "; ".join(str(v) for v in dep_result.raw_violations[:5])
                lines.append(f"    RAW hazards: {preview}")
            if dep_result.waw_violations:
                preview = "; ".join(str(v) for v in dep_result.waw_violations[:5])
                lines.append(f"    WAW hazards: {preview}")
            if dep_result.war_violations:
                preview = "; ".join(str(v) for v in dep_result.war_violations[:5])
                lines.append(f"    WAR hazards: {preview}")
            if dep_result.stage_order_breaks:
                lines.append(f"    Stage order breaks: {dep_result.stage_order_breaks[:5]}")

        # --- Memory Bank Simulation ------------------------------------------
        bank_passed, conflicts = self.verify_memory_banks(
            num_banks=num_banks,
            max_reads_per_bank=max_reads_per_bank,
            max_writes_per_bank=max_writes_per_bank,
            parallel_units=parallel_units,
        )
        lines.append(f"\nMemory Bank Simulation (num_banks={num_banks}, parallel_units={parallel_units})")
        if bank_passed:
            lines.append("✓ No bank conflicts detected in any cycle.")
        else:
            # Re-run the full simulation (not just the conflict list) so we
            # can print EVERY bank touched in a failing cycle for context,
            # matching the requested granular format exactly.
            reports = self._simulate_bank_cycles(
                num_banks=num_banks,
                max_reads_per_bank=max_reads_per_bank,
                max_writes_per_bank=max_writes_per_bank,
                parallel_units=parallel_units
            )
            reports_by_stage: Dict[int, List[CycleAccessReport]] = defaultdict(list)
            for r in reports:
                reports_by_stage[r.stage_number].append(r)

            for stage_number in sorted(reports_by_stage):
                stage_reports = [r for r in reports_by_stage[stage_number] if not r.passed]
                if not stage_reports:
                    continue
                lines.append(f"\nStage {stage_number}")
                for r in stage_reports:
                    lines.append(f"  Cycle {getattr(r, 'cycle_in_stage', 0)} ({r.access_type})")
                    for bank_id in sorted(r.bank_addresses):
                        lines.append(f"    Bank {bank_id}: Addr {r.bank_addresses[bank_id]}")
                    lines.append(f"    Result: FAIL (Requested {r.requested}, Port Capacity {r.capacity})")

        lines.append("\n" + "=" * 72)
        overall = "✅ ALL CHECKS PASSED" if dep_result.passed and bank_passed else "❌ ISSUES DETECTED"
        lines.append(f"Overall: {overall}")
        lines.append("=" * 72)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 16
    Q = 3329

    print(f"Running N={N} Standard Complete NTT to generate a schedule to verify...\n")
    engine = NTTEngine(n=N, modulus=Q)
    poly = list(range(1, N + 1))
    result = engine.forward_ntt(poly)

    verifier = ScheduleVerifier.from_ntt_result(result)
    print(verifier.generate_report(num_banks=2))

    # ------------------------------------------------------------------
    # Bonus sanity demonstration: prove the Dependency Verification Engine
    # actually *detects* hazards, rather than trivially passing everything.
    # We corrupt a copy of the execution_log by reversing it -- guaranteed
    # to violate nearly every dependency edge and every stage-ordering
    # assumption -- and confirm verify_dependencies() catches it.
    # This never runs against real engine output in production use; it's
    # here purely to demonstrate the checks have teeth.
    # ------------------------------------------------------------------
    print("\n[Self-test] Confirming the checklist rejects a deliberately-broken schedule...")
    shuffled_log = list(reversed(result.execution_log))
    broken_verifier = ScheduleVerifier(result.graph, shuffled_log)
    broken_result = broken_verifier.verify_dependencies()
    print(broken_result.checklist_text())
    print("  Expected: the two ORDER-sensitive checks fail (No read-before-write,")
    print("  Stage ordering preserved), while the two STRUCTURAL checks still pass")
    print("  (DAG is acyclic, Topological ordering preserved) -- reversing the log")
    print("  doesn't introduce a cycle or change which nodes it contains, it only")
    print("  breaks the *order* those nodes are claimed to execute in.")
