"""
NTTVis - Phase 1: Pure Python NTT Engine
==========================================

Architectural contract for this module:

1. DECOUPLED FROM GUI
   Nothing in this file imports matplotlib, tkinter, PyQt, etc. The engine's
   only job is to *compute* the NTT and *record* every butterfly operation
   into a NetworkX DAG + a linear execution log. Rendering is someone else's
   problem (see visualize_graph.py for a proof-of-concept renderer).

2. STANDARDIZED GRAPH FORMAT
   Every node in the returned `networkx.DiGraph` represents exactly one
   butterfly operation and carries these attributes:
       - stage_number    : int, which NTT stage (0-indexed) produced this node
       - butterfly_index : int, unique id of the butterfly within its stage
       - twiddle_value   : int, the twiddle factor (w) used by this butterfly
       - inputs          : tuple(int, int), memory addresses READ
       - outputs         : tuple(int, int), memory addresses WRITTEN
   Extra (non-contractual) attributes are also attached — value_in/value_out,
   op_type — because Phase 2+ (step-through debugging, animations) will want
   them. Extend, don't rename, the contractual five above.

3. EXTENSIBILITY
   The algorithm-specific logic lives entirely inside `forward_ntt()` and its
   two small helpers (`_bit_reverse_permutation`, `_find_primitive_nth_root`).
   Additional algorithm variants (see `forward_mlkem_ntt()` below for
   Kyber's 7-stage forward incomplete NTT) reuse `_new_butterfly_node()` /
   `_record_step()` so the graph format stays uniform across variants.

4. PHASE 3 -- ALGORITHM VARIATIONS
   `forward_mlkem_ntt()` implements Kyber/ML-KEM's incomplete forward NTT.
   Following the standard, this uses Cooley-Tukey (CT) Decimation-In-Time (DIT)
   butterflies, takes natural-order input, and outputs bit-reversed-like order.
   By stopping one stage early (7 stages for N=256), we leave degree-2 blocks
   (128 blocks of 2 coefficients each).
   This is followed by `mlkem_base_multiplication()`, which performs the degree-2
   base multiplication (squaring as a placeholder).
   Finally, `inverse_mlkem_ntt()` completes the pipeline using Gentleman-Sande (GS)
   Decimation-In-Frequency (DIF) inverse butterflies to restore the coefficients
   to natural order, followed by the scaling step.
   The unified pipeline is orchestrated via `run_full_mlkem_pipeline()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

import networkx as nx


# ---------------------------------------------------------------------------
# Modular-arithmetic helpers
# ---------------------------------------------------------------------------
# These are deliberately free functions (not methods) since they're generic
# number theory, not NTT-specific. Kyber/ML-KEM's real modulus q=3329 is used
# here as the "dummy" prime per Phase 1 spec — it's a real PQC prime, just
# not yet wired to any real Kyber parameter set.

def _factorize(n: int) -> List[int]:
    """Return the distinct prime factors of n (trial division is fine here;
    q-1 for our toy primes is small)."""
    factors = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    return factors


def find_generator(q: int) -> int:
    """Find a generator (primitive root) of the multiplicative group Z_q^*.

    Assumes q is prime. Works by trying small candidates g and checking that
    g^((q-1)/p) != 1 mod q for every prime factor p of (q-1); if that holds
    for all factors, g has full order (q-1) and is a generator.
    """
    phi = q - 1
    prime_factors = _factorize(phi)
    for g in range(2, q):
        if all(pow(g, phi // p, q) != 1 for p in prime_factors):
            return g
    raise ValueError(f"No generator found for q={q} (is q prime?)")


def find_primitive_nth_root(n: int, q: int, generator: Optional[int] = None) -> int:
    """Find a primitive n-th root of unity mod q, i.e. root such that
    root^n == 1 (mod q) and root^(n/2) != 1 (mod q).

    Requires n | (q-1). This is what the NTT literature calls 'omega'.
    """
    if (q - 1) % n != 0:
        raise ValueError(f"n={n} must divide q-1={q - 1} for an n-th root to exist")
    g = generator if generator is not None else find_generator(q)
    root = pow(g, (q - 1) // n, q)
    # Sanity check: this must genuinely be a PRIMITIVE n-th root, not a root
    # of some smaller order that happens to divide n.
    assert pow(root, n, q) == 1, "root^n != 1 -- generator/order math is wrong"
    if n > 1:
        assert pow(root, n // 2, q) != 1, "root has order < n, not primitive"
    return root


# ---------------------------------------------------------------------------
# Execution log entry (one per butterfly) -- plain data, GUI-friendly
# ---------------------------------------------------------------------------

@dataclass
class ButterflyStep:
    """One recorded butterfly operation. Mirrors the graph node attributes,
    but as a flat, ordered list (execution_log) which is much easier for a
    future GUI to scrub through frame-by-frame than walking the DAG."""
    node_id: str
    stage_number: int
    butterfly_index: int
    twiddle_value: int
    inputs: Tuple[int, ...]
    outputs: Tuple[int, ...]
    value_in: Tuple[int, int]     # (a[i], a[j]) BEFORE this butterfly
    value_out: Tuple[int, int]    # (a[i], a[j]) AFTER this butterfly
    op_type: str = "CT_DIT_BUTTERFLY"
    twiddle_index: int = 0
    address_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Tracks the exact count of computational operations used in the pipeline."""
    total_multiplications: int = 0
    total_additions: int = 0
    total_subtractions: int = 0


@dataclass
class NTTResult:
    """Everything Phase 2 (the GUI layer) needs, bundled together. The engine
    hands this back and does not care what happens to it afterwards."""
    output: List[int]
    graph: nx.DiGraph
    execution_log: List[ButterflyStep] = field(default_factory=list)
    bit_reversed_input: List[int] = field(default_factory=list)
    n: int = 0
    modulus: int = 0
    root_of_unity: int = 0
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class NTTEngine:
    """Pure-compute Radix-2 Cooley-Tukey (Decimation-In-Time) forward NTT
    engine that logs every butterfly into a NetworkX DAG.

    Parameters
    ----------
    n : int
        Transform length. Must be a power of two (Phase 1 restricts to the
        classic full radix-2 case; incomplete NTTs come later).
    modulus : int
        Prime modulus q. Default is Kyber's q=3329, used here purely as a
        convenient real-world PQC prime for plumbing purposes.
    root_of_unity : Optional[int]
        Primitive n-th root of unity mod q. If not supplied, one is derived
        automatically from a discovered generator of Z_q^*.
    """

    def __init__(self, n: int, modulus: int = 3329, root_of_unity: Optional[int] = None):
        if n & (n - 1) != 0 or n < 2:
            raise ValueError(f"n={n} must be a power of two >= 2")
        self.n = n
        self.q = modulus
        self.log_n = int(math.log2(n))
        self.root = root_of_unity if root_of_unity is not None else find_primitive_nth_root(n, modulus)

        # Reset per-run state. Kept as instance attributes (rather than only
        # local variables inside forward_ntt) so subclasses / callers can
        # inspect partial state, and so the pattern generalizes cleanly when
        # inverse_ntt() / incomplete-NTT variants are added later.
        self.graph: nx.DiGraph = nx.DiGraph()
        self.execution_log: List[ButterflyStep] = []
        # last_writer[address] = node_id of the most recent butterfly that
        # wrote to that memory slot. Used to wire up DAG edges correctly.
        self._last_writer: Dict[int, Optional[str]] = {}

        # Computational operations counters
        self.total_multiplications = 0
        self.total_additions = 0
        self.total_subtractions = 0

    # -- internal helpers ---------------------------------------------------

    def _bit_reverse_permutation(self) -> List[int]:
        """Return the bit-reversal permutation indices for length n.
        e.g. for n=8: [0, 4, 2, 6, 1, 5, 3, 7]
        """
        bits = self.log_n
        return [int(f"{i:0{bits}b}"[::-1], 2) for i in range(self.n)]

    def _new_butterfly_node(
        self,
        stage: int,
        butterfly_index: int,
        twiddle: int,
        inputs: Tuple[int, ...],
        outputs: Tuple[int, ...],
        val_a_before: int,
        val_b_before: int,
        val_a_after: int,
        val_b_after: int,
        twiddle_index: int,
        address_params: Dict[str, Any],
        op_type: str = "CT_DIT_BUTTERFLY",
        node_prefix: str = "B",
    ) -> str:
        """Create one operation node, attach the standardized attributes,
        wire up its incoming edges from whichever nodes last wrote to the
        memory addresses it reads, and update the write-tracker.

        Kept as its own method (rather than inlined in forward_ntt) so
        every algorithm variant -- DIT butterflies, DIF butterflies, and
        now BASE_MUL nodes -- shares one node/edge-creation code path and
        the graph format stays uniform no matter which op_type produced it.

        Parameters
        ----------
        op_type : str
            "CT_DIT_BUTTERFLY" (default), "GS_DIF_BUTTERFLY", or "BASE_MUL".
            Purely descriptive metadata -- doesn't change graph-wiring logic.
        node_prefix : str
            Distinguishes node-id namespaces within a stage, e.g. "B" for
            butterflies vs "M" for base multiplications, so a node id is
            never ambiguous even when both op_types share a stage_number.
        """
        node_id = f"S{stage}_{node_prefix}{butterfly_index}"

        self.graph.add_node(
            node_id,
            stage_number=stage,
            butterfly_index=butterfly_index,
            twiddle_value=twiddle,
            twiddle_index=twiddle_index,
            address_params=address_params,
            inputs=inputs,
            outputs=outputs,
            value_in=(val_a_before, val_b_before),
            value_out=(val_a_after, val_b_after),
            op_type=op_type,
        )

        # Wire DAG edges: this node depends on whatever last wrote to each
        # of its input addresses. If nothing did (e.g. stage 0 reading
        # the raw input directly), there's no producer node to link to --
        # this node is a source node in the DAG.
        for addr in inputs:
            producer = self._last_writer.get(addr)
            if producer is not None:
                self.graph.add_edge(producer, node_id, memory_address=addr)

        # This node is now the most recent writer of all its output addresses.
        for addr in outputs:
            self._last_writer[addr] = node_id

        return node_id

    def _record_step(self, node_id: str, stage: int, butterfly_index: int,
                      twiddle: int, inputs: Tuple[int, ...], outputs: Tuple[int, ...],
                      val_a_before: int, val_b_before: int,
                      val_a_after: int, val_b_after: int,
                      twiddle_index: int,
                      address_params: Dict[str, Any],
                      op_type: str = "CT_DIT_BUTTERFLY") -> None:
        """Append a flat ButterflyStep for easy linear GUI playback."""
        self.execution_log.append(ButterflyStep(
            node_id=node_id,
            stage_number=stage,
            butterfly_index=butterfly_index,
            twiddle_value=twiddle,
            twiddle_index=twiddle_index,
            address_params=address_params,
            inputs=inputs,
            outputs=outputs,
            value_in=(val_a_before, val_b_before),
            value_out=(val_a_after, val_b_after),
            op_type=op_type,
        ))

    # -- public API -----------------------------------------------------

    def forward_ntt(self, input_poly: List[int]) -> NTTResult:
        """Run a standard iterative Radix-2 Cooley-Tukey DIT forward NTT.

        Layout: input is bit-reversal-permuted first, then log2(n) stages of
        butterflies combine it into natural-order NTT output. This is the
        textbook in-place FFT/NTT structure.

        Butterfly (Cooley-Tukey DIT form):
            u = a[i]
            t = w * a[j]  (mod q)
            a[i] = u + t  (mod q)
            a[j] = u - t  (mod q)
        """
        if len(input_poly) != self.n:
            raise ValueError(f"input_poly length {len(input_poly)} != n={self.n}")

        # Fresh state for this run (engine instances are reusable).
        self.graph = nx.DiGraph()
        self.execution_log = []
        self._last_writer = {addr: None for addr in range(self.n)}
        self.total_multiplications = 0
        self.total_additions = 0
        self.total_subtractions = 0

        # Step 1: bit-reversal permutation of the input into working memory.
        perm = self._bit_reverse_permutation()
        a = [input_poly[perm[i]] % self.q for i in range(self.n)]
        bit_reversed_input = list(a)

        # Step 2: log2(n) butterfly stages.
        for stage in range(self.log_n):
            m = 2 ** (stage + 1)          # size of the sub-transform combined at this stage
            half_m = m // 2
            # Twiddle "step" generator for this stage: an m-th root of unity.
            w_m = pow(self.root, self.n // m, self.q)

            butterfly_index = 0
            for k in range(0, self.n, m):
                w = 1  # w_m^0
                for j in range(half_m):
                    idx_a = k + j
                    idx_b = k + j + half_m

                    val_a_before, val_b_before = a[idx_a], a[idx_b]
                    t = (w * val_b_before) % self.q
                    val_a_after = (val_a_before + t) % self.q
                    val_b_after = (val_a_before - t) % self.q
                    a[idx_a], a[idx_b] = val_a_after, val_b_after

                    self.total_multiplications += 1
                    self.total_additions += 1
                    self.total_subtractions += 1

                    twiddle_index = (self.n // m) * j
                    address_params = {
                        "stage": stage,
                        "block_start": k,
                        "offset": j,
                        "half_m": half_m
                    }

                    node_id = self._new_butterfly_node(
                        stage, butterfly_index, w, (idx_a, idx_b), (idx_a, idx_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                    )
                    self._record_step(
                        node_id, stage, butterfly_index, w, (idx_a, idx_b), (idx_a, idx_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                    )

                    butterfly_index += 1
                    w = (w * w_m) % self.q

        return NTTResult(
            output=a,
            graph=self.graph,
            execution_log=self.execution_log,
            bit_reversed_input=bit_reversed_input,
            n=self.n,
            modulus=self.q,
            root_of_unity=self.root,
            performance_metrics=PerformanceMetrics(
                total_multiplications=self.total_multiplications,
                total_additions=self.total_additions,
                total_subtractions=self.total_subtractions
            )
        )

    # -- Phase 3: Kyber's incomplete NTT --------------------------------

    def _base_mul(self, a0: int, a1: int, b0: int, b1: int, zeta: int) -> Tuple[int, int]:
        """Degree-2 base multiplication: multiplies two linear polynomials
        (a0 + a1*X) and (b0 + b1*X) modulo the quadratic (X^2 - zeta),
        which is exactly the irreducible factor left over by the incomplete
        NTT. This is the textbook Kyber `basemul` formula:

            r0 = a0*b0 + zeta*(a1*b1)
            r1 = a0*b1 + a1*b0

        Kept as its own method since it's pure arithmetic, independent of
        graph/logging concerns, and easy to unit-test in isolation.
        """
        r0 = (a0 * b0 + zeta * (a1 * b1)) % self.q
        r1 = (a0 * b1 + a1 * b0) % self.q
        return r0, r1

    def _base_mul_schoolbook(self, a0: int, a1: int, b0: int, b1: int, zeta: int) -> Tuple[int, int]:
        """Standard schoolbook multiplication using 4 multiplications:
        m0 = a0 * b0
        m1 = a1 * b1
        m2 = a0 * b1
        m3 = a1 * b0
        r0 = m0 + zeta * m1
        r1 = m2 + m3
        """
        m0 = (a0 * b0) % self.q
        m1 = (a1 * b1) % self.q
        m2 = (a0 * b1) % self.q
        m3 = (a1 * b0) % self.q
        
        r0 = (m0 + zeta * m1) % self.q
        r1 = (m2 + m3) % self.q
        return r0, r1

    def _base_mul_karatsuba(self, a0: int, a1: int, b0: int, b1: int, zeta: int) -> Tuple[int, int]:
        """Karatsuba multiplication using 3 multiplications:
        m0 = a0 * b0
        m1 = a1 * b1
        m2 = (a0 + a1) * (b0 + b1)
        r0 = m0 + zeta * m1
        r1 = m2 - m0 - m1
        """
        m0 = (a0 * b0) % self.q
        m1 = (a1 * b1) % self.q
        m2 = ((a0 + a1) * (b0 + b1)) % self.q
        
        r0 = (m0 + zeta * m1) % self.q
        r1 = (m2 - m0 - m1) % self.q
        return r0, r1

    def _forward_mlkem_ntt_core(self, input_poly: List[int], addr_offset: int = 0, node_prefix: str = "B") -> List[int]:
        """Core implementation of the 7-stage incomplete NTT.
        Allows setting an address offset and a node prefix to avoid name collisions in the DAG.
        """
        a = [x % self.q for x in input_poly]
        incomplete_stages = self.log_n - 1  # 7 for Kyber's real n=256
        k = 1
        num_bits = self.log_n - 1

        for stage in range(incomplete_stages):
            curr_len = self.n >> (stage + 1)  # block size: 128, 64, 32, 16, 8, 4, 2
            butterfly_index = 0
            
            for start in range(0, self.n, 2 * curr_len):
                # Calculate zeta for this block using the (log_n - 1)-bit bit reversal of k
                br_k = int(f"{k:0{num_bits}b}"[::-1], 2)
                zeta = pow(self.root, br_k, self.q)
                k_val = k
                k += 1

                for j in range(start, start + curr_len):
                    idx_a = j
                    idx_b = j + curr_len

                    addr_a = idx_a + addr_offset
                    addr_b = idx_b + addr_offset

                    val_a_before, val_b_before = a[idx_a], a[idx_b]
                    t = (zeta * val_b_before) % self.q
                    val_a_after = (val_a_before + t) % self.q
                    val_b_after = (val_a_before - t) % self.q
                    a[idx_a], a[idx_b] = val_a_after, val_b_after

                    self.total_multiplications += 1
                    self.total_additions += 1
                    self.total_subtractions += 1

                    twiddle_index = br_k
                    address_params = {
                        "start": start,
                        "offset": j - start,
                        "curr_len": curr_len,
                        "addr_offset": addr_offset
                    }

                    node_id = self._new_butterfly_node(
                        stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY", node_prefix=node_prefix,
                    )
                    self._record_step(
                        node_id, stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY",
                    )

                    butterfly_index += 1
        return a

    def forward_mlkem_ntt(self, input_poly: List[int]) -> NTTResult:
        """Kyber/ML-KEM-style forward NTT using Cooley-Tukey (CT) Decimation-In-Time (DIT).

        Layout: Input is in natural order, and the log2(n) - 1 stages of
        butterflies combine it in-place. For Kyber's real n=256, log2(256)-1 = 7 stages.
        Stopping it one stage early leaves degree-2 blocks (each block has 2 coefficients).
        The output coefficients are in bit-reversed-like order.

        Butterfly (Cooley-Tukey DIT form):
            t = f[j + len] * zeta   (mod q)
            f[j + len] = f[j] - t   (mod q)
            f[j] = f[j] + t         (mod q)
        """
        if len(input_poly) != self.n:
            raise ValueError(f"input_poly length {len(input_poly)} != n={self.n}")
        if self.log_n < 2:
            raise ValueError(
                f"n={self.n} is too small for an incomplete NTT (need n >= 4 "
                f"so at least one butterfly stage plus base-mul blocks exist)"
            )

        # Fresh state for this run.
        self.graph = nx.DiGraph()
        self.execution_log = []
        self._last_writer = {addr: None for addr in range(self.n)}
        self.total_multiplications = 0
        self.total_additions = 0
        self.total_subtractions = 0

        a = self._forward_mlkem_ntt_core(input_poly, addr_offset=0, node_prefix="B")

        return NTTResult(
            output=a,
            graph=self.graph,
            execution_log=self.execution_log,
            bit_reversed_input=[x % self.q for x in input_poly],  # initial natural order
            n=self.n,
            modulus=self.q,
            root_of_unity=self.root,
            performance_metrics=PerformanceMetrics(
                total_multiplications=self.total_multiplications,
                total_additions=self.total_additions,
                total_subtractions=self.total_subtractions
            )
        )

    def mlkem_base_multiplication(self, a: List[int], b: Optional[List[int]] = None, stage: Optional[int] = None, mode: str = "schoolbook") -> List[int]:
        """Degree-2 base multiplication on the outputs of two forward NTTs (a and b).

        This method iterates over the n/2 degree-2 blocks (at addresses 2i, 2i+1)
        and applies the base multiplication in either "schoolbook" mode (4 multiplications)
        or "karatsuba" mode (3 multiplications) to multiply the two polynomials,
        and logs it.

        Parameters
        ----------
        a : List[int]
            The first NTT-domain coefficient array of length n (from Polynomial A).
        b : Optional[List[int]]
            The second NTT-domain coefficient array of length n (from Polynomial B).
            If not provided, defaults to squaring a (b = a).
        stage : Optional[int]
            The stage index in the unified pipeline. Defaults to self.log_n - 1.
        mode : str
            "schoolbook" (default) or "karatsuba".
        """
        if mode not in ("schoolbook", "karatsuba"):
            raise ValueError(f"Unknown mode: {mode}")

        if stage is None:
            stage = self.log_n - 1

        if b is None:
            b = a

        # Track up to 2*n addresses if two distinct polynomials are used, otherwise self.n
        max_addr = 2 * self.n if b is not a else self.n
        if not self._last_writer:
            self._last_writer = {addr: None for addr in range(max_addr)}

        num_bits = self.log_n - 1
        op_type = "BASE_MUL_SCHOOLBOOK" if mode == "schoolbook" else "BASE_MUL_KARATSUBA"

        c = [0] * self.n

        for i in range(self.n // 2):
            idx0, idx1 = 2 * i, 2 * i + 1
            a0, a1 = a[idx0], a[idx1]
            b0, b1 = b[idx0], b[idx1]

            # zeta: root^(2 * bit_reverse(i) + 1)
            br_i = int(f"{i:0{num_bits}b}"[::-1], 2)
            zeta = pow(self.root, 2 * br_i + 1, self.q)

            if mode == "schoolbook":
                r0, r1 = self._base_mul_schoolbook(a0, a1, b0, b1, zeta)
                self.total_multiplications += 4
                self.total_additions += 1
                self.total_subtractions += 2
            else:
                r0, r1 = self._base_mul_karatsuba(a0, a1, b0, b1, zeta)
                self.total_multiplications += 3
                self.total_additions += 3
                self.total_subtractions += 2

            c[idx0], c[idx1] = r0, r1

            # Create node in graph
            node_id = f"S{stage}_M{i}"
            
            # If b is a, inputs are idx0, idx1. If b is distinct, inputs are idx0, idx1 and offset idx0+n, idx1+n
            if b is not a:
                inputs_tuple = (idx0, idx1, idx0 + self.n, idx1 + self.n)
                value_in_tuple = (a0, a1) # Display values of Polynomial A in active step
            else:
                inputs_tuple = (idx0, idx1)
                value_in_tuple = (a0, a1)

            twiddle_index = 2 * br_i + 1
            address_params = {
                "i": i,
                "is_distinct": b is not a,
                "n": self.n
            }

            self.graph.add_node(
                node_id,
                stage_number=stage,
                butterfly_index=i,
                twiddle_value=zeta,
                twiddle_index=twiddle_index,
                address_params=address_params,
                inputs=inputs_tuple,
                outputs=(idx0, idx1),
                value_in=value_in_tuple,
                value_out=(r0, r1),
                op_type=op_type,
            )

            # Wire DAG edges from the last writers of inputs
            for addr in inputs_tuple:
                producer = self._last_writer.get(addr)
                if producer is not None:
                    self.graph.add_edge(producer, node_id, memory_address=addr)

            # Set this node as the last writer of C's addresses idx0 and idx1
            self._last_writer[idx0] = node_id
            self._last_writer[idx1] = node_id

            self._record_step(
                node_id, stage, i, zeta, inputs_tuple, (idx0, idx1),
                a0, a1, r0, r1,  # logs values in (for slider display compatibility)
                twiddle_index=twiddle_index,
                address_params=address_params,
                op_type=op_type,
            )

        return c

    def inverse_mlkem_ntt(self, a: List[int], start_stage: Optional[int] = None) -> List[int]:
        """Kyber/ML-KEM-style inverse NTT using Gentleman-Sande (GS) Decimation-In-Frequency (DIF).

        This takes the bit-reversed input from the base multiplication, runs the 7 inverse
        stages to restore it to natural order, and applies scaling by (N/2)^-1 mod q.

        Butterfly (Gentleman-Sande DIF form):
            t = f[j]
            f[j] = t + f[j + len]               (mod q)
            f[j + len] = zeta * (f[j + len] - t) (mod q)

        Parameters
        ----------
        a : List[int]
            The coefficient array of length n.
        start_stage : Optional[int]
            The stage index in the unified pipeline. Defaults to 0.
        """
        if start_stage is None:
            start_stage = 0

        if not self._last_writer:
            self._last_writer = {addr: None for addr in range(self.n)}

        k = self.n // 2 - 1
        num_bits = self.log_n - 1

        for s in range(self.log_n - 1):
            curr_len = 2 ** (s + 1)  # 2, 4, 8, ..., 128
            stage_num = start_stage + s
            butterfly_index = 0

            for start in range(0, self.n, 2 * curr_len):
                # Calculate zeta for this block using the (log_n - 1)-bit bit reversal of k
                br_k = int(f"{k:0{num_bits}b}"[::-1], 2)
                zeta = pow(self.root, br_k, self.q)
                k_val = k
                k -= 1

                for j in range(start, start + curr_len):
                    idx_a = j
                    idx_b = j + curr_len

                    val_a_before, val_b_before = a[idx_a], a[idx_b]
                    val_a_after = (val_a_before + val_b_before) % self.q
                    val_b_after = (zeta * (val_b_before - val_a_before)) % self.q
                    a[idx_a], a[idx_b] = val_a_after, val_b_after

                    self.total_multiplications += 1
                    self.total_additions += 1
                    self.total_subtractions += 1

                    twiddle_index = br_k
                    address_params = {
                        "start": start,
                        "offset": j - start,
                        "curr_len": curr_len
                    }

                    node_id = self._new_butterfly_node(
                        stage_num, butterfly_index, zeta, (idx_a, idx_b), (idx_a, idx_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="GS_DIF_BUTTERFLY", node_prefix="B",
                    )
                    self._record_step(
                        node_id, stage_num, butterfly_index, zeta, (idx_a, idx_b), (idx_a, idx_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="GS_DIF_BUTTERFLY",
                    )

                    butterfly_index += 1

        # Scaling step at the very end
        inv_n_2 = pow(self.n // 2, self.q - 2, self.q)
        for i in range(self.n):
            a[i] = (a[i] * inv_n_2) % self.q

        return a

    def run_full_mlkem_pipeline(self, poly_a: List[int], poly_b: List[int], mode: str = "schoolbook") -> NTTResult:
        """Run the full, mathematically accurate ML-KEM NTT pipeline for two polynomials C = INTT(NTT(A) o NTT(B)).

        This chains:
        1. Forward CT NTT on poly_a and poly_b stage-by-stage (A uses addresses 0..N-1 prefix A_, B uses N..2N-1 prefix B_).
        2. Base Multiplication (schoolbook or karatsuba, outputs to 0..N-1).
        3. Inverse GS NTT on the output array (restores 0..N-1 to natural order).

        Returns a single NTTResult containing the combined output, DAG, and execution log.
        """
        if len(poly_a) != self.n or len(poly_b) != self.n:
            raise ValueError(f"Polynomial lengths must match n={self.n}")

        # Fresh state for this run.
        self.graph = nx.DiGraph()
        self.execution_log = []
        self._last_writer = {addr: None for addr in range(2 * self.n)}
        self.total_multiplications = 0
        self.total_additions = 0
        self.total_subtractions = 0

        a = [x % self.q for x in poly_a]
        b = [x % self.q for x in poly_b]

        incomplete_stages = self.log_n - 1  # 7 for Kyber's real n=256
        num_bits = self.log_n - 1

        # Run forward NTT stage-by-stage concurrently for A and B to preserve stage ordering
        for stage in range(incomplete_stages):
            curr_len = self.n >> (stage + 1)  # block size: 128, 64, 32, 16, 8, 4, 2
            
            # --- Polynomial A stage s ---
            k = 2 ** stage
            butterfly_index = 0
            for start in range(0, self.n, 2 * curr_len):
                br_k = int(f"{k:0{num_bits}b}"[::-1], 2)
                zeta = pow(self.root, br_k, self.q)
                k_val = k
                k += 1

                for j in range(start, start + curr_len):
                    idx_a = j
                    idx_b = j + curr_len
                    addr_a = idx_a
                    addr_b = idx_b

                    val_a_before, val_b_before = a[idx_a], a[idx_b]
                    t = (zeta * val_b_before) % self.q
                    val_a_after = (val_a_before + t) % self.q
                    val_b_after = (val_a_before - t) % self.q
                    a[idx_a], a[idx_b] = val_a_after, val_b_after

                    self.total_multiplications += 1
                    self.total_additions += 1
                    self.total_subtractions += 1

                    twiddle_index = br_k
                    address_params = {
                        "start": start,
                        "offset": j - start,
                        "curr_len": curr_len,
                        "addr_offset": 0
                    }

                    node_id = self._new_butterfly_node(
                        stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY", node_prefix="A_",
                    )
                    self._record_step(
                        node_id, stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY",
                    )
                    butterfly_index += 1

            # --- Polynomial B stage s ---
            k = 2 ** stage
            butterfly_index = 0
            for start in range(0, self.n, 2 * curr_len):
                br_k = int(f"{k:0{num_bits}b}"[::-1], 2)
                zeta = pow(self.root, br_k, self.q)
                k_val = k
                k += 1

                for j in range(start, start + curr_len):
                    idx_a = j
                    idx_b = j + curr_len
                    addr_a = idx_a + self.n
                    addr_b = idx_b + self.n

                    val_a_before, val_b_before = b[idx_a], b[idx_b]
                    t = (zeta * val_b_before) % self.q
                    val_a_after = (val_a_before + t) % self.q
                    val_b_after = (val_a_before - t) % self.q
                    b[idx_a], b[idx_b] = val_a_after, val_b_after

                    self.total_multiplications += 1
                    self.total_additions += 1
                    self.total_subtractions += 1

                    twiddle_index = br_k
                    address_params = {
                        "start": start,
                        "offset": j - start,
                        "curr_len": curr_len,
                        "addr_offset": self.n
                    }

                    node_id = self._new_butterfly_node(
                        stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY", node_prefix="B_",
                    )
                    self._record_step(
                        node_id, stage, butterfly_index, zeta, (addr_a, addr_b), (addr_a, addr_b),
                        val_a_before, val_b_before, val_a_after, val_b_after,
                        twiddle_index=twiddle_index,
                        address_params=address_params,
                        op_type="CT_DIT_BUTTERFLY",
                    )
                    butterfly_index += 1

        # Step 3: Base Multiplication (at stage 7 for N=256)
        c = self.mlkem_base_multiplication(a, b, stage=self.log_n - 1, mode=mode)

        # Step 4: Inverse NTT (from stage 8 to 14 for N=256)
        c = self.inverse_mlkem_ntt(c, start_stage=self.log_n)

        # Combined initial input is size 2N (poly_a followed by poly_b) to avoid IndexErrors in replay
        bit_reversed_input = [x % self.q for x in poly_a] + [x % self.q for x in poly_b]

        return NTTResult(
            output=c,
            graph=self.graph,
            execution_log=self.execution_log,
            bit_reversed_input=bit_reversed_input,
            n=self.n,
            modulus=self.q,
            root_of_unity=self.root,
            performance_metrics=PerformanceMetrics(
                total_multiplications=self.total_multiplications,
                total_additions=self.total_additions,
                total_subtractions=self.total_subtractions
            )
        )


# ---------------------------------------------------------------------------
# Naive O(n^2) reference NTT -- used only to sanity-check the fast engine.
# ---------------------------------------------------------------------------

def naive_ntt(input_poly: List[int], n: int, q: int, root: int) -> List[int]:
    return [
        sum(input_poly[j] * pow(root, i * j, q) for j in range(n)) % q
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Memory Conflict / Collision calculation helper (used by verifiers / UI)
# ---------------------------------------------------------------------------

def calculate_memory_bank_collisions(
    execution_log: List[Any],
    num_banks: int,
    banking_mode: str,
    parallel_units: int,
    n: int = 256
) -> Tuple[bool, List[Any], int, List[Any]]:
    """Exposes memory bank collision logic from the verifier.
    Returns (passed, conflicts, total_architectural_bottlenecks, reports).
    """
    from verification_engine import ScheduleVerifier
    verifier = ScheduleVerifier(None, execution_log, n=n)
    reports = verifier._simulate_bank_cycles(
        num_banks=num_banks,
        max_reads_per_bank=2,
        max_writes_per_bank=2,
        banking_mode=banking_mode,
        parallel_units=parallel_units
    )
    passed, conflicts = verifier.verify_memory_banks(
        num_banks=num_banks,
        max_reads_per_bank=2,
        max_writes_per_bank=2,
        banking_mode=banking_mode,
        parallel_units=parallel_units
    )
    total_bottlenecks = sum(max(0, c.requested_accesses - c.port_capacity) for c in conflicts)
    return passed, conflicts, total_bottlenecks, reports


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 8
    Q = 3329  # Kyber's prime, used here just as a convenient dummy modulus

    engine = NTTEngine(n=N, modulus=Q)
    print(f"N={N}, q={Q}, primitive {N}-th root of unity = {engine.root}")

    poly = [1, 2, 3, 4, 5, 6, 7, 8]
    result = engine.forward_ntt(poly)

    print(f"\nInput polynomial:            {poly}")
    print(f"Bit-reversed input:           {result.bit_reversed_input}")
    print(f"Fast NTT output:              {result.output}")

    reference = naive_ntt(poly, N, Q, engine.root)
    print(f"Naive O(n^2) reference NTT:   {reference}")
    print(f"Match: {result.output == reference}")

    print(f"\nDAG summary: {result.graph.number_of_nodes()} nodes "
          f"(butterflies), {result.graph.number_of_edges()} edges (dependencies)")
    print(f"Execution log has {len(result.execution_log)} recorded steps")

    print("\nFirst node's attributes (proof of standardized schema):")
    first_node = list(result.graph.nodes(data=True))[0]
    print(f"  id: {first_node[0]}")
    for k, v in first_node[1].items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 3 demo: Kyber-style full ML-KEM NTT pipeline demo, using n=16
    # here just to keep console output short.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 3: Full ML-KEM NTT Pipeline Demo (Schoolbook)")
    print("=" * 60)

    engine16 = NTTEngine(n=16, modulus=Q)
    poly16_a = list(range(1, 17))
    poly16_b = [i * 2 for i in range(1, 17)]
    pipeline_result = engine16.run_full_mlkem_pipeline(poly16_a, poly16_b, mode="schoolbook")

    expected_fwd_stages = engine16.log_n - 1
    expected_inv_stages = engine16.log_n - 1
    expected_base_muls = engine16.n // 2
    
    op_types = {step.op_type for step in pipeline_result.execution_log}
    ct_butterfly_count = sum(1 for s in pipeline_result.execution_log if s.op_type == "CT_DIT_BUTTERFLY")
    gs_butterfly_count = sum(1 for s in pipeline_result.execution_log if s.op_type == "GS_DIF_BUTTERFLY")
    base_mul_count = sum(1 for s in pipeline_result.execution_log if s.op_type == "BASE_MUL_SCHOOLBOOK")

    print(f"N=16 -> expected {expected_fwd_stages} forward stages (ran twice), 1 base_mul stage, {expected_inv_stages} inverse stages")
    print(f"Recorded op_types: {op_types}")
    print(f"CT Butterfly steps: {ct_butterfly_count} (expect {expected_fwd_stages * 16})")
    print(f"BASE_MUL steps:     {base_mul_count} (expect {expected_base_muls})")
    print(f"GS Butterfly steps: {gs_butterfly_count} (expect {expected_inv_stages * 8})")
    print(f"Pipeline output:    {pipeline_result.output}")

    print("\n" + "=" * 60)
    print("Phase 3: Full ML-KEM NTT Pipeline Demo (Karatsuba)")
    print("=" * 60)

    pipeline_result_k = engine16.run_full_mlkem_pipeline(poly16_a, poly16_b, mode="karatsuba")
    op_types_k = {step.op_type for step in pipeline_result_k.execution_log}
    base_mul_count_k = sum(1 for s in pipeline_result_k.execution_log if s.op_type == "BASE_MUL_KARATSUBA")
    print(f"Recorded op_types: {op_types_k}")
    print(f"BASE_MUL steps:     {base_mul_count_k} (expect {expected_base_muls})")
    print(f"Pipeline output matches Schoolbook: {pipeline_result.output == pipeline_result_k.output}")
