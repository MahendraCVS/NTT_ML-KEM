"""
NTTVis - Phase 3: Interactive Streamlit Dashboard (+ Algorithm Variations)
=============================================================================

This app is a pure PRESENTATION layer -- all NTT math lives in
`ntt_engine.NTTEngine`. This file's job is to:
    1. Let the user configure a run (N, input polynomial, and NOW which
       algorithm variant to compute -- standard complete NTT or Kyber's
       incomplete NTT).
    2. Call the engine once per configuration (cached).
    3. Let the user "time-travel" through the recorded execution_log with a
       slider, showing one recorded step (butterfly OR base multiplication)
       at a time.
    4. Render the DAG with the active node/edges highlighted, past nodes
       shown normally, future nodes dimmed -- and, new in Phase 3, BASE_MUL
       nodes drawn as squares so they're visually distinct from the
       circular butterfly nodes.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import pandas as pd
import streamlit as st
import numpy as np

from ntt_engine import NTTEngine, NTTResult, ButterflyStep, naive_ntt
from verification_engine import ScheduleVerifier, CycleAccessReport
from reference_validator import ReferenceValidator

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
MODULUS = 3329          # Kyber's prime -- used as a convenient dummy q
# 256 is included because Kyber's incomplete NTT is only "the real thing"
# (exactly 7 stages, 128 base multiplications) at n=256. Smaller values
# still run correctly (fewer stages/base-muls) and are handy for a
# quick-to-render, pedagogical view of the DAG structure.
DEGREE_OPTIONS = [8, 16, 32, 64, 128, 256]

ALGO_STANDARD = "Standard Complete NTT"
ALGO_INCOMPLETE = "Kyber Incomplete NTT (7-Stage)"

# Above this many DAG nodes, per-node text labels get dropped (still fully
# color/shape coded) so the plot stays fast and legible rather than a wall
# of overlapping text -- this mainly bites at N=256.
LABEL_NODE_THRESHOLD = 150

# Hardcoded NIST / Reference test vector (dummy data structured for easy hex swap later)
NIST_VECTOR_N = 256
NIST_VECTOR_A = list(range(256))
NIST_VECTOR_B = [i * 2 for i in range(256)]
NIST_VECTOR_EXPECTED = [400, 2212, 1211, 730, 773, 1344, 2447, 757, 2936, 2330, 2272, 2766, 487, 2097, 942, 355, 340, 901, 2042, 438, 2751, 2327, 2499, 3271, 1318, 3302, 2569, 2452, 2955, 753, 2508, 1566, 1260, 1594, 2572, 869, 3147, 2752, 3017, 617, 2214, 1154, 770, 1066, 2046, 385, 2745, 2472, 2899, 701, 2540, 1762, 1700, 2358, 411, 2521, 2034, 2283, 3272, 1676, 828, 732, 1392, 2812, 1667, 1290, 1685, 2856, 1478, 884, 1078, 2064, 517, 3099, 3156, 692, 2369, 1533, 1517, 2325, 632, 3100, 3075, 561, 2220, 1398, 1428, 2314, 731, 12, 161, 1182, 3079, 2527, 2859, 750, 2862, 2541, 3120, 1274, 336, 310, 1200, 3010, 2415, 2748, 684, 2885, 2697, 124, 1828, 1155, 1438, 2681, 1559, 1405, 2223, 688, 133, 562, 1979, 1059, 1135, 2211, 962, 721, 1492, 3279, 2757, 3259, 1460, 693, 962, 2271, 1295, 1367, 2491, 1342, 1253, 2228, 942, 728, 1590, 203, 3229, 685, 2562, 2206, 2950, 1469, 1096, 1835, 361, 7, 777, 2675, 2376, 3213, 1861, 1653, 2593, 1356, 1275, 2354, 1268, 1350, 2604, 1705, 1986, 122, 2775, 3291, 1674, 1257, 2044, 710, 588, 1682, 667, 876, 2313, 1653, 2229, 716, 447, 1426, 328, 486, 1904, 1257, 1878, 442, 282, 1402, 477, 840, 2495, 2117, 3039, 1936, 2141, 329, 3162, 657, 2805, 2952, 1102, 588, 1414, 255, 444, 1985, 1553, 2481, 1444, 1775, 149, 3228, 1029, 214, 787, 2752, 2784, 887, 394, 1309, 307, 721, 2555, 2484, 512, 3301, 868, 3204, 326, 2225, 2247, 396, 5, 1078, 290, 974, 3134, 116, 1911, 1865, 3311, 2924, 708, 3325, 792, 3100, 266, 2281, 2491, 900]

st.set_page_config(page_title="ML-KEM Architectural and Execution Analytics & Architectural Verification Framework", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# Engine invocation -- cached so we only recompute when config changes
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running NTT engine and building execution trace...")
def run_ntt_cached(n: int, modulus: int, poly_a: tuple[int, ...], poly_b: tuple[int, ...], algorithm: str, base_mul_mode: str = "schoolbook") -> NTTResult:
    """Thin, cached wrapper around the Phase 1/3 engine.

    `algorithm` and `base_mul_mode` are part of the cache key alongside (n, modulus, poly_a, poly_b),
    so switching the dropdown correctly triggers a fresh computation while
    switching back to a previously-seen configuration hits the cache again
    -- no redundant recomputation either way.
    """
    engine = NTTEngine(n=n, modulus=modulus)
    if algorithm == ALGO_INCOMPLETE:
        return engine.run_full_mlkem_pipeline(list(poly_a), list(poly_b), mode=base_mul_mode)
    return engine.forward_ntt(list(poly_a))


def generate_random_poly(n: int, modulus: int) -> list[int]:
    """Uniform-random coefficients in [0, modulus)."""
    return [random.randint(0, modulus - 1) for _ in range(n)]


@st.cache_data(show_spinner="Computing golden reference (naive O(N²) NTT)...")
def compute_naive_reference_cached(n: int, modulus: int, poly: tuple[int, ...], root: int) -> list[int]:
    """Cached wrapper around the brute-force reference NTT from ntt_engine.py.

    This is a completely independent computation path from the fast
    Cooley-Tukey engine (no shared code beyond the modulus/root), which is
    exactly what makes agreement between the two meaningful as a
    correctness check. Cached on (n, modulus, poly, root) so it doesn't
    re-run on every slider drag -- only when the underlying config changes.
    """
    return naive_ntt(list(poly), n, modulus, root)


def naive_poly_mul(poly_a: list[int], poly_b: list[int], modulus: int) -> list[int]:
    """Brute-force polynomial multiplication in Z_q[X]/(X^N + 1)."""
    n = len(poly_a)
    c = [0] * n
    for i in range(n):
        for j in range(n):
            idx = i + j
            coef = (poly_a[i] * poly_b[j]) % modulus
            if idx >= n:
                c[idx - n] = (c[idx - n] - coef) % modulus
            else:
                c[idx] = (c[idx] + coef) % modulus
    return c


@st.cache_data(show_spinner="Computing golden reference (naive ring polynomial multiplication)...")
def compute_naive_poly_mul_cached(poly_a: tuple[int, ...], poly_b: tuple[int, ...], modulus: int) -> list[int]:
    return naive_poly_mul(list(poly_a), list(poly_b), modulus)


def draw_bank_heatmap(reports: list[CycleAccessReport], num_banks: int, capacity: int = 2) -> plt.Figure:
    """Generate a Matplotlib heatmap showing memory bank accesses and conflicts per cycle/stage."""
    # Sort reports: stage first, then Reads, then Writes
    sorted_reports = sorted(reports, key=lambda r: (r.stage_number, 0 if r.access_type == "Reads" else 1))
    
    y_labels = [f"Stage {r.stage_number} ({r.access_type})" for r in sorted_reports]
    
    data_grid = []
    for r in sorted_reports:
        row = [len(r.bank_addresses.get(b, [])) for b in range(num_banks)]
        data_grid.append(row)
        
    data = np.array(data_grid)
    
    # Figure sizing based on bank count and stages count
    fig, ax = plt.subplots(figsize=(max(5, num_banks * 0.9), max(3, len(y_labels) * 0.35)))
    
    if data.size > 0:
        vmax = max(4, np.max(data))
    else:
        vmax = 4
        
    # Use YlOrRd colormap to naturally transition from light yellow (no conflict) to deep red (high conflicts)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
    
    # Set grid ticks and labels
    ax.set_xticks(np.arange(num_banks))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels([f"Bank {b}" for b in range(num_banks)], fontsize=8.5)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    
    # Render value count inside each cell
    for i in range(len(y_labels)):
        for j in range(num_banks):
            val = data[i, j]
            # White text for high access counts (darker red background) for visibility
            text_color = "white" if val > capacity else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")
            
    ax.set_title("Memory Access Heat Map", fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

def replay_array_state(bit_reversed_input: list[int], execution_log: list[ButterflyStep], up_to_step: int) -> list[int]:
    """Reconstruct the working-memory array state step by step."""
    array = list(bit_reversed_input)
    for step in execution_log[: up_to_step + 1]:
        addr_a, addr_b = step.outputs
        val_a, val_b = step.value_out
        if 0 <= addr_a < len(array):
            array[addr_a] = val_a
        if 0 <= addr_b < len(array):
            array[addr_b] = val_b
    return array

def build_trace_dataframe(execution_log: list[ButterflyStep]) -> pd.DataFrame:
    """Flatten execution_log into the exact column layout execution analysts
    need to feed into their own simulators: one row per operation, in
    execution order, with every address/twiddle/op-type field spelled out.
    """
    return pd.DataFrame([
        {
            "Stage": step.stage_number,
            "Op Type": step.op_type,
            "Node ID": step.node_id,
            "Twiddle/Base Root": step.twiddle_value,
            "Input Addr A": step.inputs[0],
            "Input Addr B": step.inputs[1],
            "Output Addr A": step.outputs[0],
            "Output Addr B": step.outputs[1],
        }
        for step in execution_log
    ])


# ---------------------------------------------------------------------------
# Matplotlib DAG renderer -- highlights execution progress at `current_step`
# ---------------------------------------------------------------------------

def draw_ntt_dag(
    graph: nx.DiGraph,
    execution_log: list[ButterflyStep],
    current_step: int,
    n: int,
    modulus: int,
) -> plt.Figure:
    """Render the DAG with three visual STATES (active / executed / pending)
    crossed with two visual SHAPES (butterfly = circle, BASE_MUL = square).

    States (color/alpha):
        - ACTIVE   (the op at `current_step`): bright red, drawn on top.
        - EXECUTED (steps 0..current_step-1): normal stage-colored, solid.
        - PENDING  (steps not yet reached): dimmed gray, low alpha.
    Shapes (op_type):
        - CT_DIT_BUTTERFLY / GS_DIF_BUTTERFLY -> circle ('o')
        - BASE_MUL                            -> square ('s')

    Layout is stable across reruns (multipartite by stage_number) so the
    graph doesn't jump around as the user scrubs the slider -- only
    colors/shapes-per-node-group change frame to frame.
    """
    active_step = execution_log[current_step]
    active_node = active_step.node_id
    executed_nodes = {step.node_id for step in execution_log[: current_step + 1]}

    pos = nx.multipartite_layout(graph, subset_key="stage_number")
    num_stages = max(d["stage_number"] for _, d in graph.nodes(data=True)) + 1
    stage_cmap = plt.get_cmap("viridis")

    total_nodes = graph.number_of_nodes()
    show_labels = total_nodes <= LABEL_NODE_THRESHOLD
    node_size = 1000 if show_labels else max(120, 45000 // total_nodes)

    fig, ax = plt.subplots(figsize=(2.6 + 2.2 * min(num_stages, 10), 5.5))

    # --- Partition nodes into shape groups (butterfly vs base-mul) -------
    square_nodes = {nd for nd, d in graph.nodes(data=True) if d["op_type"] in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA")}
    circle_nodes = set(graph.nodes) - square_nodes

    pending = {nd for nd in graph.nodes if nd not in executed_nodes}
    executed_not_active = executed_nodes - {active_node}

    def stage_colors(nodelist):
        return [stage_cmap(graph.nodes[nd]["stage_number"] / max(num_stages - 1, 1))
                for nd in nodelist]

    # Draw order: pending (background) -> executed (mid) -> active (top),
    # and within each state, circles then squares, so both shapes are
    # visible in every visual state.
    for shape_set, marker in ((circle_nodes, "o"), (square_nodes, "s")):
        pending_list = list(pending & shape_set)
        if pending_list:
            nx.draw_networkx_nodes(graph, pos, nodelist=pending_list, node_shape=marker,
                                    node_color="lightgray", alpha=0.35,
                                    edgecolors="gray", node_size=node_size * 0.8, ax=ax)

    for shape_set, marker in ((circle_nodes, "o"), (square_nodes, "s")):
        exec_list = list(executed_not_active & shape_set)
        if exec_list:
            nx.draw_networkx_nodes(graph, pos, nodelist=exec_list, node_shape=marker,
                                    node_color=stage_colors(exec_list), alpha=0.95,
                                    edgecolors="black", node_size=node_size, ax=ax)

    active_marker = "s" if active_node in square_nodes else "o"
    nx.draw_networkx_nodes(graph, pos, nodelist=[active_node], node_shape=active_marker,
                            node_color="#FF4B4B", alpha=1.0, edgecolors="black",
                            linewidths=2.0, node_size=node_size * 1.3, ax=ax)

    # --- Edges: same active/executed/pending split as before -------------
    incoming_to_active = set(graph.in_edges(active_node))
    executed_edges = [
        (u, v) for u, v in graph.edges
        if v in executed_nodes and (u, v) not in incoming_to_active
    ]
    pending_edges = [(u, v) for u, v in graph.edges if v not in executed_nodes]

    nx.draw_networkx_edges(graph, pos, edgelist=pending_edges, edge_color="lightgray",
                            alpha=0.3, arrows=True, arrowsize=10, ax=ax,
                            connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(graph, pos, edgelist=executed_edges, edge_color="gray",
                            alpha=0.8, arrows=True, arrowsize=12, width=1.2, ax=ax,
                            connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(graph, pos, edgelist=list(incoming_to_active), edge_color="#FF8C00",
                            alpha=1.0, arrows=True, arrowsize=16, width=2.5, ax=ax,
                            connectionstyle="arc3,rad=0.05")

    # --- Labels (skipped for large graphs -- see LABEL_NODE_THRESHOLD) ---
    if show_labels:
        labels = {}
        for node, d in graph.nodes(data=True):
            if d["op_type"] in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA"):
                labels[node] = f"M{d['butterfly_index']}\nζ={d['twiddle_value']}"
            else:
                labels[node] = f"B{d['butterfly_index']}\nw={d['twiddle_value']}"
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7.5)

    # --- Legend explaining the shape encoding -----------------------------
    legend_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=9, label="Butterfly"),
        mlines.Line2D([], [], color="gray", marker="s", linestyle="None",
                      markersize=9, label="Base Multiplication (BASE_MUL)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=8)

    ax.set_title(
        f"NTT DAG (N={n}, q={modulus})  |  Step {current_step + 1}/{len(execution_log)}  |  "
        f"Active: Stage {active_step.stage_number}, "
        f"{'BaseMul' if active_step.op_type in ('BASE_MUL', 'BASE_MUL_SCHOOLBOOK', 'BASE_MUL_KARATSUBA') else 'Butterfly'} "
        f"{active_step.butterfly_index}"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Execution Framework")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Verification Oracle")
uploaded_file = st.sidebar.file_uploader(
    "Upload Official ML-KEM Reference Dump (.txt, .json)",
    type=["txt", "json"],
    help="Upload a C-printf dump or JSON file from the official ML-KEM reference implementation."
)

if uploaded_file is not None:
    try:
        file_content = uploaded_file.getvalue().decode("utf-8")
        validator = ReferenceValidator()
        parsed_a, parsed_b, parsed_expected = validator.parse_reference_file(file_content)
        n = len(parsed_a)
        algorithm = ALGO_INCOMPLETE
        
        # Keep base multiplication strategy selection in sidebar
        base_mul_strategy = st.sidebar.selectbox(
            "Base Multiplication Strategy",
            ["Schoolbook (4 Multiplications)", "Karatsuba (3 Multiplications)"],
            help="Karatsuba reduces coefficient-coefficient multiplications from 4 to 3.",
            key="uploaded_base_mul"
        )
        base_mul_mode = "schoolbook" if "Schoolbook" in base_mul_strategy else "karatsuba"
        
        st.session_state.poly_a = list(parsed_a)
        st.session_state.poly_b = list(parsed_b)
        st.session_state.poly_n = n
        st.session_state.parsed_expected = list(parsed_expected)
        
        st.sidebar.success("✅ Uploaded file parsed successfully!")
        st.sidebar.info(f"Official ML-KEM Reference Integration Mode Active: N={n}, Incomplete NTT.")
        load_nist_vector = False
    except Exception as e:
        st.sidebar.error(f"Failed to parse reference file: {e}")
        load_nist_vector = False
        n = 256
        algorithm = ALGO_INCOMPLETE
        base_mul_mode = "schoolbook"
else:
    load_nist_vector = st.sidebar.checkbox(
        "Load Standard ML-KEM Test Vector",
        value=False,
        help="Forces N=256 and loads the hardcoded NIST reference test vector."
    )

    if load_nist_vector:
        n = 256
        algorithm = ALGO_INCOMPLETE
        base_mul_mode = "schoolbook"
        st.session_state.poly_a = list(NIST_VECTOR_A)
        st.session_state.poly_b = list(NIST_VECTOR_B)
        st.session_state.poly_n = 256
        st.sidebar.info("NIST Test Vector Mode Active: N=256, Incomplete NTT, Schoolbook Strategy.")
    else:
        algorithm = st.sidebar.selectbox(
            "Algorithm Selection",
            [ALGO_STANDARD, ALGO_INCOMPLETE],
            help=(
                "Standard: full log2(N)-stage Cooley-Tukey NTT, resolves every "
                "coefficient. Kyber Incomplete: stops log2(N)-1 stages early "
                "(exactly 7 stages at N=256, matching real Kyber/ML-KEM), leaving "
                "N/2 degree-2 blocks that get a BASE_MUL step instead."
            ),
        )

        base_mul_mode = "schoolbook"
        if algorithm == ALGO_INCOMPLETE:
            base_mul_strategy = st.sidebar.selectbox(
                "Base Multiplication Strategy",
                ["Schoolbook (4 Multiplications)", "Karatsuba (3 Multiplications)"],
                help="Karatsuba reduces coefficient-coefficient multiplications from 4 to 3.",
            )
            base_mul_mode = "schoolbook" if "Schoolbook" in base_mul_strategy else "karatsuba"

        n = st.sidebar.selectbox("Polynomial Degree (N)", DEGREE_OPTIONS, index=0)

if algorithm == ALGO_INCOMPLETE and n < 8:
    st.sidebar.warning("N must be >= 8 for a meaningful incomplete NTT (need >=2 DIF stages).")

if n == 256 and not load_nist_vector and uploaded_file is None:
    st.sidebar.info(
        "N=256 is the real Kyber/ML-KEM size. Node labels are hidden above "
        f"{LABEL_NODE_THRESHOLD} nodes to keep the graph readable; color and "
        "shape coding still fully reflect execution state."
    )

# Keep polynomials A and B in session_state so they survive reruns without regenerating.
# Reset to default polynomials whenever N changes (only when not loading custom files).
if uploaded_file is None and not load_nist_vector:
    if "poly_a" not in st.session_state or st.session_state.get("poly_n") != n:
        st.session_state.poly_a = list(range(1, n + 1))  # simple deterministic default for A
        st.session_state.poly_b = [i * 2 for i in range(1, n + 1)]  # simple deterministic default for B
        st.session_state.poly_n = n

    input_method = st.sidebar.radio(
        "Input Method",
        ["Random Generation", "Manual Input"],
        help="Choose whether to generate random polynomials or input coefficients manually."
    )

    if input_method == "Random Generation":
        if st.sidebar.button("🎲 Generate Random Polynomials"):
            st.session_state.poly_a = generate_random_poly(n, MODULUS)
            st.session_state.poly_b = generate_random_poly(n, MODULUS)
            st.session_state.poly_n = n
    else:
        default_val_a = ", ".join(map(str, st.session_state.poly_a))
        default_val_b = ", ".join(map(str, st.session_state.poly_b))
        
        raw_poly_a = st.sidebar.text_area(
            "Polynomial A Coefficients",
            value=default_val_a,
            help=f"Enter {n} comma-separated integers for Polynomial A.",
            placeholder="e.g. 12, -45, 3329, 0..."
        )
        raw_poly_b = st.sidebar.text_area(
            "Polynomial B Coefficients",
            value=default_val_b,
            help=f"Enter {n} comma-separated integers for Polynomial B.",
            placeholder="e.g. 12, -45, 3329, 0..."
        )
        
        if st.sidebar.button("Load Manual Polynomials"):
            def parse_poly_str(raw_str: str, label: str) -> list[int]:
                cleaned = raw_str.strip().strip("[]").strip()
                if not cleaned:
                    raise ValueError(f"{label} is empty.")
                parts = cleaned.split(",")
                parsed_vals = []
                for idx, p in enumerate(parts):
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    try:
                        parsed_vals.append(int(p_clean))
                    except ValueError:
                        raise ValueError(f"Invalid integer '{p_clean}' in {label} at position {idx+1}.")
                return parsed_vals

            try:
                parsed_a = parse_poly_str(raw_poly_a, "Polynomial A")
                parsed_b = parse_poly_str(raw_poly_b, "Polynomial B")
                
                if len(parsed_a) != n:
                    st.sidebar.error(f"Error: Expected {n} coefficients for Polynomial A, but got {len(parsed_a)}.")
                    st.stop()
                if len(parsed_b) != n:
                    st.sidebar.error(f"Error: Expected {n} coefficients for Polynomial B, but got {len(parsed_b)}.")
                    st.stop()
                
                normalized_a = [coeff % MODULUS for coeff in parsed_a]
                normalized_b = [coeff % MODULUS for coeff in parsed_b]
                
                st.session_state.poly_a = normalized_a
                st.session_state.poly_b = normalized_b
                st.session_state.poly_n = n
                st.sidebar.success("Manual polynomials loaded and normalized successfully!")
            except ValueError as ve:
                st.sidebar.error(f"Parsing Error: {ve}")
                st.stop()

st.sidebar.caption(f"Modulus q = {MODULUS} (fixed, Kyber prime)")
with st.sidebar.expander("Current Input Polynomial A"):
    st.write(st.session_state.poly_a)

with st.sidebar.expander("Current Input Polynomial B"):
    st.write(st.session_state.poly_b)

st.sidebar.markdown("**🛡️ Execution Verification Model**")
num_banks = st.sidebar.slider(
    "Number of Memory Banks",
    min_value=1,
    max_value=8,
    value=2,
    help=(
        "Simulated dual-port RAM banks. Drive this up "
        "to see how many banks are actually needed for a conflict-free "
        "schedule at the current N."
    ),
)
banking_strategy = st.sidebar.selectbox(
    "Memory Banking Strategy",
    ["Modulo", "Block", "XOR"],
    help="Select the address mapping function used to partition memory into banks."
)
banking_mode = banking_strategy.lower()

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Verification Settings")
enable_fault_injection = st.sidebar.checkbox(
    "⚠️ Enable Fault Injection Mode (For Verifier Testing)",
    value=False,
    help="Deliberately mutates the execution log right before it hits the validator to test its detection capabilities."
)

fault_type = ""
if enable_fault_injection:
    fault_type = st.sidebar.selectbox(
        "Select Fault to Inject",
        [
            "Swap Execution Order (Schedule Violation)",
            "Corrupt Twiddle Index (Twiddle Mismatch)",
            "Flip Output Address (Address Mismatch / Hazard)",
            "Corrupt Final Output (Functional Mismatch)"
        ],
        help="Choose which type of error to inject into the execution trace."
    )

# ---------------------------------------------------------------------------
# Run the (cached) engine
# ---------------------------------------------------------------------------

poly_a_tuple = tuple(st.session_state.poly_a)
poly_b_tuple = tuple(st.session_state.poly_b)
result: NTTResult = run_ntt_cached(n, MODULUS, poly_a_tuple, poly_b_tuple, algorithm, base_mul_mode)

# Initialize session state for Fault Injection Lab
if "faulted_result" not in st.session_state:
    st.session_state.faulted_result = None
if "active_fault_config" not in st.session_state:
    st.session_state.active_fault_config = None

# Sidebar: Fault Injection Lab
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Fault Injection Lab")

fault_lab_type = st.sidebar.selectbox(
    "Fault Lab Action",
    ["No Fault", "Zeroize Stage", "Zeroize Pointer"],
    help="Select the fault type from Ravi et al. (2022). Zeroize Stage zeroizes twiddles for all nodes at a stage; Zeroize Pointer zeroizes twiddles for a specific module."
)

if fault_lab_type != "No Fault":
    available_stages = sorted(list(set(step.stage_number for step in result.execution_log)))
    fault_stage = st.sidebar.selectbox("Stage to Fault", available_stages, key="fault_stage_selectbox")
    
    has_multiple_modules = any("A_" in step.node_id or "B_" in step.node_id for step in result.execution_log)
    fault_module = None
    if fault_lab_type == "Zeroize Pointer" and has_multiple_modules:
        fault_module = st.sidebar.selectbox("Module Index to Fault", [0, 1], format_func=lambda x: f"Module {x} (Forward NTTs)" if x == 0 else f"Module {x} (Inverse NTT)")
    
    if st.sidebar.button("Run Fault Injection"):
        from fault_engine import FaultInjector
        if fault_lab_type == "Zeroize Stage":
            st.session_state.faulted_result = FaultInjector.zeroize_stage(result, fault_stage)
        elif fault_lab_type == "Zeroize Pointer":
            st.session_state.faulted_result = FaultInjector.zeroize_pointer(result, fault_stage, fault_module)
        
        st.session_state.active_fault_config = {
            "n": n,
            "algorithm": algorithm,
            "poly_a": list(st.session_state.poly_a),
            "poly_b": list(st.session_state.poly_b),
            "base_mul_mode": base_mul_mode,
            "fault_type": fault_lab_type,
            "fault_stage": fault_stage,
            "fault_module": fault_module
        }
        st.sidebar.success("⚡ Fault injected successfully!")

# Check if config has changed, or if No Fault is selected, clear fault state
config_changed = False
if st.session_state.active_fault_config is not None:
    cfg = st.session_state.active_fault_config
    has_multiple_modules = any("A_" in step.node_id or "B_" in step.node_id for step in result.execution_log)
    fault_stage_val = st.session_state.get("fault_stage_selectbox", None)
    if (cfg.get("n") != n or 
        cfg.get("algorithm") != algorithm or 
        cfg.get("poly_a") != list(st.session_state.poly_a) or 
        cfg.get("poly_b") != list(st.session_state.poly_b) or 
        cfg.get("base_mul_mode") != base_mul_mode or
        cfg.get("fault_type") != fault_lab_type or
        (fault_lab_type != "No Fault" and cfg.get("fault_stage") != fault_stage_val) or
        (fault_lab_type == "Zeroize Pointer" and has_multiple_modules and cfg.get("fault_module") != (st.session_state.get("fault_module") if "fault_module" in st.session_state else None))):
        config_changed = True

if config_changed or fault_lab_type == "No Fault":
    st.session_state.faulted_result = None
    st.session_state.active_fault_config = None

if st.session_state.faulted_result is not None:
    active_result = st.session_state.faulted_result
else:
    active_result = result

# Create copies of log, graph and output to support fault injection without corrupting cache
import copy
mutated_log = copy.deepcopy(active_result.execution_log)
mutated_graph = active_result.graph.copy()
mutated_output = list(active_result.output)

injected_fault_desc = ""

if enable_fault_injection and len(mutated_log) > 0:
    if fault_type == "Swap Execution Order (Schedule Violation)":
        swapped = False
        for i in range(1, len(mutated_log)):
            node_id = mutated_log[i].node_id
            preds = list(mutated_graph.predecessors(node_id))
            if preds:
                # Find predecessor node in log
                pred_id = preds[0]
                pred_idx = -1
                for idx, step in enumerate(mutated_log):
                    if step.node_id == pred_id:
                        pred_idx = idx
                        break
                if pred_idx != -1 and pred_idx < i:
                    # Swap them!
                    mutated_log[pred_idx], mutated_log[i] = mutated_log[i], mutated_log[pred_idx]
                    injected_fault_desc = f"Swapped execution order of '{mutated_log[pred_idx].node_id}' and '{mutated_log[i].node_id}'."
                    swapped = True
                    break
        if not swapped and len(mutated_log) >= 2:
            mutated_log[0], mutated_log[1] = mutated_log[1], mutated_log[0]
            injected_fault_desc = f"Swapped execution order of first two nodes: '{mutated_log[0].node_id}' and '{mutated_log[1].node_id}'."
            
    elif fault_type == "Corrupt Twiddle Index (Twiddle Mismatch)":
        corrupted = False
        for step in mutated_log:
            if step.op_type in ("CT_DIT_BUTTERFLY", "GS_DIF_BUTTERFLY"):
                old_val = step.twiddle_value
                step.twiddle_value = (step.twiddle_value + 13) % MODULUS
                if step.node_id in mutated_graph:
                    mutated_graph.nodes[step.node_id]["twiddle_value"] = step.twiddle_value
                injected_fault_desc = f"Corrupted twiddle value of '{step.node_id}' from {old_val} to {step.twiddle_value}."
                corrupted = True
                break
        if not corrupted:
            step = mutated_log[0]
            old_val = step.twiddle_value
            step.twiddle_value = (step.twiddle_value + 13) % MODULUS
            if step.node_id in mutated_graph:
                mutated_graph.nodes[step.node_id]["twiddle_value"] = step.twiddle_value
            injected_fault_desc = f"Corrupted twiddle value of '{step.node_id}' from {old_val} to {step.twiddle_value}."

    elif fault_type == "Flip Output Address (Address Mismatch / Hazard)":
        step = mutated_log[0]
        old_outputs = step.outputs
        step.outputs = (9999, step.outputs[1])
        if step.node_id in mutated_graph:
            mutated_graph.nodes[step.node_id]["outputs"] = step.outputs
        injected_fault_desc = f"Flipped output address of '{step.node_id}' from {old_outputs} to {step.outputs}."

    elif fault_type == "Corrupt Final Output (Functional Mismatch)":
        old_val = mutated_output[0]
        mutated_output[0] = (mutated_output[0] + 1) % MODULUS
        injected_fault_desc = f"Corrupted index 0 of final output polynomial from {old_val} to {mutated_output[0]}."

execution_log = mutated_log
max_step = len(execution_log) - 1

# Run the verifier ONCE per rerun using the possibly mutated copies
verifier = ScheduleVerifier(mutated_graph, execution_log, n=n)
verification = verifier.verify_all(num_banks=num_banks, banking_mode=banking_mode)
dag_statistics = verification["dag_statistics"]

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("ML-KEM NTT Architectural and Execution Analytics & Instance Verifier")
st.subheader("Visualize, verify, benchmark, and diagnose cryptographic execution and architectural analytics pipelines.")
if uploaded_file is not None:
    st.caption("Official ML-KEM Reference Integration — step through every recorded operation.")
else:
    st.caption(f"{algorithm} — step through every recorded operation.")

# --- Time-travel slider ----------------------------------------------------
st.subheader("⏱️ Time-Travel Through Execution")
current_step = st.slider(
    "Execution step",
    min_value=0,
    max_value=max_step,
    value=0,
    help="Scrub through every recorded operation (butterflies, and base "
         "multiplications where applicable), in execution order.",
    # IMPORTANT: without an explicit key, Streamlit treats `value=0` as a
    # one-time initial default and thereafter restores whatever value the
    # user last set for this widget -- even after N or algorithm changes
    # make that value out of range for the *new* execution_log. Keying on
    # (n, algorithm) forces a brand-new widget (reset to step 0) whenever
    # the configuration changes, instead of a stale/confusing leftover step.
    key=f"step_slider_{n}_{algorithm}_{base_mul_mode}",
)
active: ButterflyStep = execution_log[current_step]
is_base_mul = active.op_type in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA")

# --- Active state display ---------------------------------------------------
st.subheader("🔎 Active Operation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stage", active.stage_number)
c2.metric("Base-Mul Index" if is_base_mul else "Butterfly Index", active.butterfly_index)
c3.metric("Zeta (ζ)" if is_base_mul else "Twiddle Factor (w)", active.twiddle_value)
c4.metric("Op Type", active.op_type)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Input Addr A", active.inputs[0])
c6.metric("Input Addr B", active.inputs[1])
c7.metric("Output Addr A", active.outputs[0])
c8.metric("Output Addr B", active.outputs[1])

c9, c10 = st.columns(2)
with c9:
    st.markdown("**Values IN** (before)")
    st.code(f"a[{active.inputs[0]}] = {active.value_in[0]}\n"
            f"a[{active.inputs[1]}] = {active.value_in[1]}")
with c10:
    st.markdown("**Values OUT** (after)")
    st.code(f"a[{active.outputs[0]}] = {active.value_out[0]}\n"
            f"a[{active.outputs[1]}] = {active.value_out[1]}")

if is_base_mul:
    st.caption(
        "This BASE_MUL step demonstrates Kyber's degree-2 pointwise "
        "multiplication (squaring the transformed polynomial against "
        "itself here), which is the operation the incomplete NTT exists "
        "to set up — see docstring in `ntt_engine.py` for the formula."
    )

# --- Graph rendering ---------------------------------------------------
if st.session_state.faulted_result is not None:
    st.subheader("🕸️ Execution DAGs Comparison")
    st.caption(
        "🔴 Active op · 🟠 Edges feeding it · Colored = already executed · "
        "Gray/dim = not yet reached · ⬤ Butterfly · ◼ Base Multiplication"
    )
    col_dag1, col_dag2 = st.columns(2)
    with col_dag1:
        st.markdown("#### 🟢 Clean Execution DAG")
        fig_clean = draw_ntt_dag(result.graph, result.execution_log, current_step, n, MODULUS)
        st.pyplot(fig_clean)
        plt.close(fig_clean)
    with col_dag2:
        st.markdown("#### 🔴 Faulted Execution DAG")
        fig_faulted = draw_ntt_dag(mutated_graph, execution_log, current_step, n, MODULUS)
        st.pyplot(fig_faulted)
        plt.close(fig_faulted)
else:
    st.subheader("🕸️ Execution DAG")
    st.caption(
        "🔴 Active op · 🟠 Edges feeding it · Colored = already executed · "
        "Gray/dim = not yet reached · ⬤ Butterfly · ◼ Base Multiplication"
    )
    fig = draw_ntt_dag(mutated_graph, execution_log, current_step, n, MODULUS)
    st.pyplot(fig)
    plt.close(fig)  # free the figure -- important in Streamlit to avoid memory buildup

# --- Optional: current memory state ----------------------------------------
with st.expander("📦 View working-memory array state at this step"):
    array_state = replay_array_state(result.bit_reversed_input, execution_log, current_step)
    st.write(array_state)

# --- Final output (always available, independent of slider) ----------------
with st.expander("✅ Final output (all recorded steps complete)"):
    if st.session_state.faulted_result is not None:
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            st.markdown("**Clean Output:**")
            st.write(result.output)
        with col_out2:
            st.markdown("**Faulted Output:**")
            st.write(mutated_output)
    else:
        st.write(mutated_output)

# ---------------------------------------------------------------------------
# Twiddle & Memory Trace
# ---------------------------------------------------------------------------
# The full execution_log as a flat table -- exactly what an execution analyst
# needs to feed into their own cycle-accurate simulator: every operation, in
# order, with its stage, twiddle/base-root, and the four memory addresses it
# touches.

st.subheader("📝 Twiddle & Memory Trace")
st.caption(
    "Every recorded operation, in execution order, with its twiddle factor "
    "(or base-multiplication root) and the memory addresses it reads/writes."
)

trace_df = build_trace_dataframe(execution_log)
st.dataframe(trace_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Trace as CSV",
    data=trace_df.to_csv(index=False),
    file_name="ntt_twiddle_trace.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Computational Cost Analysis & Execution & Architectural Analytics section
# ---------------------------------------------------------------------------

st.subheader("📊 Computational Cost Analysis & Execution & Architectural Analytics")
metrics = result.performance_metrics
c_mult, c_add, c_sub = st.columns(3)
c_mult.metric("Total Multiplications (Multiplier Count)", f"{metrics.total_multiplications:,}")
c_add.metric("Total Additions (Adder Count)", f"{metrics.total_additions:,}")
c_sub.metric("Total Subtractions (Subtractor Count)", f"{metrics.total_subtractions:,}")

if algorithm == ALGO_INCOMPLETE:
    if base_mul_mode == "karatsuba":
        st.info(f"Karatsuba Optimization Active: Saved {n // 2} multipliers (Multiplier Count) at the cost of {n} additional adders (Adder Count) compared to Schoolbook.")
    else:
        st.info("Schoolbook Base Multiplication: Highest multiplier (Multiplier Count) usage. Switch to Karatsuba to optimize resource usage.")
else:
    st.info("Standard Complete NTT: Base multiplication is not active in this pipeline.")



# ---------------------------------------------------------------------------
# Performance metrics section
# ---------------------------------------------------------------------------

st.subheader("📊 Performance Metrics")

butterfly_steps = [s for s in execution_log if s.op_type not in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA")]
base_mul_steps = [s for s in execution_log if s.op_type in ("BASE_MUL", "BASE_MUL_SCHOOLBOOK", "BASE_MUL_KARATSUBA")]

total_butterflies = len(butterfly_steps)
total_base_muls = len(base_mul_steps)
# Every recorded step -- butterfly or base-mul -- reads 2 addresses and
# writes 2 addresses, so this generalizes cleanly across algorithms.
total_memory_reads = sum(len(step.inputs) for step in execution_log)
total_memory_writes = sum(len(step.outputs) for step in execution_log)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Butterflies", total_butterflies)
m2.metric("Total Base Multiplications", total_base_muls)
m3.metric("Total Memory Reads", total_memory_reads)
m4.metric("Total Memory Writes", total_memory_writes)
m5.metric("Total DAG Stages", result.graph.nodes[execution_log[-1].node_id]["stage_number"] + 1)

st.caption("DAG structural statistics (from the Execution Instance Verification engine):")
d1, d2, d3 = st.columns(3)
d1.metric("Graph Depth (Critical Path)", f"{dag_statistics['graph_depth']} hops",
          help="Length of the longest dependency chain -- the minimum number "
               "of sequential cycles needed even with unlimited parallel execution units.")
d2.metric("Graph Width (Max Parallel)", dag_statistics['graph_width'],
          help=f"Most operations sharing a single stage (stage "
               f"{dag_statistics['widest_stage']}) -- the minimum number of "
               f"parallel execution units needed to run any one stage in one cycle.")
d3.metric("Total Dependency Edges", dag_statistics['total_edges'])

# ---------------------------------------------------------------------------
# Execution Instance Schedule Verification (Phase 4 engine, wired into the GUI)
# ---------------------------------------------------------------------------
# Re-runs on every rerun of this script -- i.e. automatically whenever N,
# algorithm, polynomial, or num_banks changes, since `result` (from the
# cached engine call above) and `num_banks` (from the sidebar) are both
# already fresh by the time we get here. No extra caching needed: this is
# just grouping + counting over an execution_log that's at most ~1000
# entries even at N=256, so it's effectively instant.

# ---------------------------------------------------------------------------
# Calculate Verification Results for Dashboard Checklist
# ---------------------------------------------------------------------------

# 1. Reference Polynomial & Functional Verification
if uploaded_file is not None:
    expected_reference_poly = list(st.session_state.parsed_expected)
elif load_nist_vector:
    expected_reference_poly = list(NIST_VECTOR_EXPECTED)
else:
    if algorithm == ALGO_STANDARD:
        expected_reference_poly = compute_naive_reference_cached(n, MODULUS, poly_a_tuple, result.root_of_unity)
    else:
        expected_reference_poly = compute_naive_poly_mul_cached(poly_a_tuple, poly_b_tuple, MODULUS)

validator = ReferenceValidator()
if uploaded_file is not None:
    ref_report = validator.verify_reference_vector(mutated_output, expected_reference_poly, reference_type="external")
else:
    ref_report = validator.verify_reference_vector(mutated_output, expected_reference_poly, reference_type="internal")

# Parse matches / total
summary_line = ref_report.splitlines()[-1]
try:
    match_part = summary_line.split("Overall: ")[1].split(" MATCH")[0]
    matches, total = map(int, match_part.split(" / "))
except Exception:
    matches, total = 0, n

functional_passed = (matches == total)
functional_msg = f"{matches} / {total} coefficients match reference."

# 2. Dependency Verification
dep_detail = verification["dependency_check"]["detail"]
# Standard dependency checks are indices 0 to 5 in dep_detail.checks
dependency_passed = all(dep_detail.checks[i][1] for i in range(min(6, len(dep_detail.checks))))
dependency_msg = "Dependency structure and hazards verified." if dependency_passed else "Dependency hazard / structure violation detected."

# 3. Schedule Verification
schedule_passed = dep_detail.schedule_error_message is None
schedule_msg = dep_detail.schedule_error_message or "All dependencies executed in correct sequence order."

# 4. Address Verification
agu_report = validator.verify_address_generation(execution_log, n)
address_passed = True
address_msg = "No duplicate, missing, or out-of-bounds address accesses."
for stage, chk in agu_report.items():
    if (chk["completeness"] != "PASS" or 
        chk["no_duplicates"] != "PASS" or 
        chk["bounds"] != "PASS" or 
        chk.get("address_math", "PASS") != "PASS"):
        address_passed = False
        details = chk["details"]
        if chk["completeness"] != "PASS":
            address_msg = f"Stage {stage} address completeness check failed."
        elif chk["no_duplicates"] != "PASS":
            address_msg = f"Stage {stage} duplicate address access."
        elif chk["bounds"] != "PASS":
            address_msg = f"Stage {stage} out-of-bounds address."
        else:
            address_msg = f"Stage {stage} mathematical address formula mismatch."
        break

# 5. Twiddle Verification
twiddle_df = validator.verify_twiddle_factors(
    execution_log,
    n=n,
    root=result.root_of_unity,
    modulus=result.modulus,
    algorithm_type=algorithm
)
if not twiddle_df.empty:
    failures = twiddle_df[twiddle_df["Result"] == "FAIL"]
    twiddle_passed = len(failures) == 0
    twiddle_msg = "All twiddle factors mathematically verified." if twiddle_passed else f"{len(failures)} twiddle factor mismatch(es) detected."
else:
    twiddle_passed = True
    twiddle_msg = "No twiddle factor operations in trace."

# 6. Memory Verification
bank_check = verification["memory_bank_check"]
memory_passed = bank_check["passed"]
memory_msg = "Memory bank access conflict-free." if memory_passed else f"{len(bank_check['conflicts'])} port conflict(s) detected."

# 7. Output Entropy Verification
entropy_res = verifier.verify_output_entropy(active_result, threshold=0.9)
entropy_passed = entropy_res.passed
if entropy_passed:
    entropy_msg = f"Output entropy {entropy_res.final_entropy:.4f} is above threshold 0.90."
else:
    anomaly_stage_str = f"stage {entropy_res.first_anomaly_stage}" if entropy_res.first_anomaly_stage is not None else "N/A"
    entropy_msg = f"Output entropy {entropy_res.final_entropy:.4f} below threshold 0.90 (first anomaly at {anomaly_stage_str})."

checklist_results = [
    ("Functional Verification", functional_passed, functional_msg),
    ("Dependency Verification", dependency_passed, dependency_msg),
    ("Schedule Verification", schedule_passed, schedule_msg),
    ("Address Verification", address_passed, address_msg),
    ("Twiddle Verification", twiddle_passed, twiddle_msg),
    ("Memory Verification", memory_passed, memory_msg),
    ("Output Entropy Verification", entropy_passed, entropy_msg),
]

st.subheader("🛡️ Execution Verification Engine Dashboard")
st.caption(
    "Strict categorized checklist verification of the execution instance: "
    "mathematical logic, dependencies, sequence scheduling, twiddle factors, "
    "address generation unit, and banked-memory ports."
)

# Render the checks in a beautiful styled card
checklist_html = """
<div style="background-color: rgba(128, 128, 128, 0.08); padding: 15px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.2); margin-bottom: 20px;">
"""
for label, passed, msg in checklist_results:
    if passed:
        checklist_html += f'<div style="color: #2e7d32; font-size: 1.1rem; font-weight: bold; margin: 8px 0;">✓ {label} <span style="font-weight: normal; font-size: 0.95rem; color: #777; margin-left: 10px;">({msg})</span></div>'
    else:
        checklist_html += f'<div style="color: #c62828; font-size: 1.1rem; font-weight: bold; margin: 8px 0;">✗ {label} <span style="font-weight: normal; font-size: 0.95rem; color: #555; margin-left: 10px;">(Violation — {msg})</span></div>'
checklist_html += "</div>"
st.markdown(checklist_html, unsafe_allow_html=True)

if enable_fault_injection:
    st.warning(f"⚠️ **Chaos Testing Mode Active**: Injected fault '{fault_type}'")
    failed_items = [label for label, passed, _ in checklist_results if not passed]
    if failed_items:
        st.success(f"🎯 **Verifier Robustness Confirmed**: The Execution Verification Engine successfully caught the injected fault: **{failed_items[0]}**!")
        for label, passed, msg in checklist_results:
            if not passed:
                st.info(f"**Caught Error Details ({label})**: {msg}")
    else:
        st.error("❌ **Verifier Fault**: Injected fault was not caught by the verification engine.")

# --- Detailed sub-sections for drill-down ---

# --- 1. Dependency & Schedule Verification Details ---
st.markdown("### 🕸️ Dependency & Sequence Check details")
with st.expander("🔬 View Detailed Dependency & Sequence Verification Log", expanded=not dependency_passed or not schedule_passed):
    st.markdown("**Detailed Sub-Checks Checklist:**")
    st.text(dep_detail.checklist_text())
    if dep_detail.schedule_error_message:
        st.error(dep_detail.schedule_error_message)
    if dep_detail.violations:
        st.write("Violations:")
        st.write(dep_detail.violations)

# --- 2. Memory Bank Simulation Details ---
st.markdown(f"### 🗄️ Memory Bank Simulation (banks = {bank_check['num_banks']}, strategy = {banking_strategy})")
st.caption(
    f"This simulation checks for port resource bottlenecks based on the selected "
    f"**{banking_strategy}** banking model. Results and conflicts are specific to this mapping."
)
if not bank_check["passed"]:
    conflicts = bank_check["conflicts"]
    st.error(
        f"{len(conflicts)} architectural bottleneck(s) detected across "
        f"{bank_check['num_banks']} bank(s) using the {banking_strategy} strategy. "
        f"Try raising 'Number of Memory Banks' or switching the 'Memory Banking Strategy' in the sidebar."
    )
    with st.expander("View Detailed Architectural Bottlenecks"):
        st.dataframe(
            [
                {
                    "Stage": c.stage_number,
                    "Bank": c.bank_id,
                    "Access Type": c.access_type,
                    "Requested": c.requested_accesses,
                    "Available Ports": c.port_capacity,
                    "Node IDs": ", ".join(c.conflicting_node_ids),
                }
                for c in conflicts
            ],
            width=True,
        )

# --- Memory Access Heat Map ---
st.markdown(f"**Memory Access Heat Map**")
st.caption(
    f"The heatmap below visualizes the total memory read/write requests mapped to each "
    f"physical bank per execution stage under the **{banking_strategy}** model. "
    f"Any bank cell with value > 2 (capacity limit) is an architectural access conflict."
)
reports = verifier._simulate_bank_cycles(
    num_banks=num_banks,
    max_reads_per_bank=2,
    max_writes_per_bank=2,
    banking_mode=banking_mode
)
fig_heatmap = draw_bank_heatmap(reports, num_banks, capacity=2)
st.pyplot(fig_heatmap)
plt.close(fig_heatmap)

# --- 3. Reference Verification Details ---
st.markdown("### 🥇 Functional Reference Verification Details")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Reference Verification Match", f"{matches} / {total} Match")
with col2:
    st.progress(matches / total)

if not functional_passed:
    st.error(f"FAIL — {total - matches} coefficient(s) disagree with the reference vector.")

with st.expander("🔬 View Detailed Coefficient Verification Report", expanded=(load_nist_vector or uploaded_file is not None or not functional_passed)):
    st.text_area(
        "Reference Validator Output Log",
        value=ref_report,
        height=300,
        disabled=True
    )

# --- 4. Twiddle Verification Details ---
st.markdown("### 🔍 Cycle-by-Cycle Twiddle Verification Details")
st.caption(
    "Mathematically reconstructs the expected twiddle factor for every stage "
    "and butterfly operation, and compares it directly against the value recorded "
    "in the execution trace."
)
if not twiddle_passed:
    st.error(f"FAIL — {len(failures)} twiddle factor mismatch(es) detected in the execution trace!")
if not twiddle_df.empty:
    st.dataframe(twiddle_df, use_container_width=True)
else:
    st.info("No twiddle factor operations to verify in the current trace.")

# --- 5. Address Verification Details ---
st.markdown("### 🛑 Address Generation Verification Details")
st.caption(
    "Proves that the AGU (Address Generation Unit) accesses every memory address "
    "exactly once per stage, ensuring completeness, no duplicate accesses, and "
    "correct bounds limits."
)
if not address_passed:
    st.error("FAIL — Address Generation issues detected in the execution trace!")

# Render a summary table of the validation status per stage
agu_data = []
for stage, chk in sorted(agu_report.items()):
    agu_data.append({
        "Stage": stage,
        "Completeness": chk["completeness"],
        "No Duplicates": chk["no_duplicates"],
        "Bounds": chk["bounds"],
        "Mathematical Check": chk.get("address_math", "PASS"),
    })
st.dataframe(pd.DataFrame(agu_data), use_container_width=True)

with st.expander("🔍 View Detailed Address Generation Verification Log", expanded=not address_passed):
    for stage, chk in sorted(agu_report.items()):
        details = chk["details"]
        is_stage_ok = (
            chk["completeness"] == "PASS" and 
            chk["no_duplicates"] == "PASS" and 
            chk["bounds"] == "PASS" and 
            chk.get("address_math", "PASS") == "PASS"
        )
        status_str = "🟢 PASS" if is_stage_ok else "🔴 FAIL"
        
        st.markdown(f"**Stage {stage} ({status_str})**")
        st.write(f"Expected bound limit: `[0, {details['expected_bound_max'] - 1}]`")
        
        if chk["completeness"] != "PASS":
            if details["missing_reads"]:
                st.write(f"  - Missing reads: {details['missing_reads']}")
            if details["missing_writes"]:
                st.write(f"  - Missing writes: {details['missing_writes']}")
                
        if chk["no_duplicates"] != "PASS":
            if details["duplicate_reads"]:
                st.write(f"  - Duplicate reads: {details['duplicate_reads']}")
            if details["duplicate_writes"]:
                st.write(f"  - Duplicate writes: {details['duplicate_writes']}")
                
        if chk["bounds"] != "PASS":
            if details["out_of_bounds_reads"]:
                st.write(f"  - Out-of-bounds reads: {details['out_of_bounds_reads']}")
            if details["out_of_bounds_writes"]:
                st.write(f"  - Out-of-bounds writes: {details['out_of_bounds_writes']}")

        if chk.get("address_math", "PASS") != "PASS":
            if details.get("stage_math_failures"):
                for fail in details["stage_math_failures"]:
                    st.write(f"  - Mathematical mismatch: {fail}")

# --- 6. Output Entropy Stage Profile ---
st.markdown("### 📊 Output Entropy Stage Profile")
st.caption(
    "Shannon entropy (normalized) calculated stage by stage over output values. "
    "Under fault injection, you will observe the entropy drop or halve stage-by-stage."
)

from fault_engine import compute_entropy_profile
clean_profile = compute_entropy_profile(result)

if st.session_state.faulted_result is not None:
    faulted_profile = compute_entropy_profile(st.session_state.faulted_result)
    
    # Render both profiles on a chart
    entropy_data = []
    for stage in sorted(clean_profile.keys()):
        entropy_data.append({
            "Stage": f"Stage {stage}",
            "Clean Entropy": clean_profile[stage],
            "Faulted Entropy": faulted_profile.get(stage, 0.0)
        })
    df_entropy = pd.DataFrame(entropy_data).set_index("Stage")
    st.bar_chart(df_entropy)
else:
    entropy_data = []
    for stage in sorted(clean_profile.keys()):
        entropy_data.append({
            "Stage": f"Stage {stage}",
            "Entropy": clean_profile[stage]
        })
    df_entropy = pd.DataFrame(entropy_data).set_index("Stage")
    st.bar_chart(df_entropy)
