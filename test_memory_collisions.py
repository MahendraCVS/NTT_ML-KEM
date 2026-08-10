import pytest
from ntt_engine import NTTEngine, calculate_memory_bank_collisions
from verification_engine import ScheduleVerifier

def test_parallel_chunking_distinguishes_strategies():
    """Verify that under cycle-accurate parallel chunking, different banking
    strategies (Modulo, Block, XOR) yield different bottleneck counts.
    """
    n = 256
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    input_poly = list(range(1, n + 1))
    
    # Run standard complete NTT
    result = engine.forward_ntt(input_poly)
    
    # Calculate collisions for different banking strategies
    _, _, bottlenecks_mod, _ = calculate_memory_bank_collisions(
        result.execution_log, num_banks=8, banking_mode="modulo", parallel_units=6, n=n
    )
    
    _, _, bottlenecks_blk, _ = calculate_memory_bank_collisions(
        result.execution_log, num_banks=8, banking_mode="block", parallel_units=6, n=n
    )
    
    _, _, bottlenecks_xor, _ = calculate_memory_bank_collisions(
        result.execution_log, num_banks=8, banking_mode="xor", parallel_units=6, n=n
    )
    
    # Verify that Modulo, Block, and XOR banking report different number of bottlenecks
    bottlenecks = [bottlenecks_mod, bottlenecks_blk, bottlenecks_xor]
    assert len(set(bottlenecks)) > 1, f"Strategies should not yield identical bottlenecks: {bottlenecks}"


def test_parallel_chunking_various_units():
    """Verify that different parallel units yield different bottleneck counts
    due to varying cycle partitioning.
    """
    n = 256
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    input_poly = list(range(1, n + 1))
    
    # Run standard complete NTT
    result = engine.forward_ntt(input_poly)
    
    # Compare parallel_units = 4 vs 8
    _, _, bottlenecks_u4, _ = calculate_memory_bank_collisions(
        result.execution_log, num_banks=4, banking_mode="modulo", parallel_units=4, n=n
    )
    
    _, _, bottlenecks_u8, _ = calculate_memory_bank_collisions(
        result.execution_log, num_banks=4, banking_mode="modulo", parallel_units=8, n=n
    )
    
    # Verify that larger parallel unit size yields higher bottleneck count
    assert bottlenecks_u4 == 0
    assert bottlenecks_u8 > bottlenecks_u4

