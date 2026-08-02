import math
import pytest

from ntt_engine import NTTEngine
from fault_engine import FaultInjector, compute_entropy_profile


def test_zeroize_stage_standard_eq5():
    """Validate that zeroize_stage on N=8 standard NTT reproduces the paper's Eq. 5 pattern.

    Under full zeroization of all stages, the output coefficients should satisfy
    output[i] == output[i - n/2] for i >= n/2.
    """
    n = 8
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    input_poly = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Run clean standard NTT
    result = engine.forward_ntt(input_poly)
    
    # Zeroize all stages (0, 1, 2)
    faulted_result = result
    for stage in range(int(math.log2(n))):
        faulted_result = FaultInjector.zeroize_stage(faulted_result, stage)
        
    # Verify the output[i] == output[i - n/2] pattern for i >= n/2
    output = faulted_result.output
    half_n = n // 2
    for i in range(half_n, n):
        assert output[i] == output[i - half_n]


def test_zeroize_stage_incomplete_eq7():
    """Validate that zeroize_stage on incomplete (Kyber) NTT reproduces Eq. 7.

    Specifically, only the first 2 coefficients survive/match the original input
    coefficients at their respective positions (output[0] == input[0] and
    output[1] == input[1]), and all other elements are repetitions of these two.
    """
    n = 8
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    input_poly = [10, 20, 30, 40, 50, 60, 70, 80]
    
    # Run clean Kyber forward NTT (incomplete)
    result = engine.forward_mlkem_ntt(input_poly)
    
    # Incomplete NTT has stages 0 and 1. Zeroize both.
    faulted_result = result
    faulted_result = FaultInjector.zeroize_stage(faulted_result, 0)
    faulted_result = FaultInjector.zeroize_stage(faulted_result, 1)
    
    output = faulted_result.output
    # The first two coefficients must equal the first two input coefficients
    assert output[0] == input_poly[0]
    assert output[1] == input_poly[1]
    
    # Check that all other coefficients are repetitions of these first two
    for i in range(n):
        if i % 2 == 0:
            assert output[i] == input_poly[0]
        else:
            assert output[i] == input_poly[1]


def test_entropy_profile_strictly_decreases():
    """Validate that entropy_profile strictly decreases stage-over-stage under full zeroization."""
    n = 8
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    input_poly = [1, 2, 3, 4, 5, 6, 7, 8]
    
    result = engine.forward_ntt(input_poly)
    
    # Zeroize all stages (0, 1, 2)
    faulted_result = result
    for stage in range(int(math.log2(n))):
        faulted_result = FaultInjector.zeroize_stage(faulted_result, stage)
        
    profile = compute_entropy_profile(faulted_result)
    
    # Ensure profile entries exist for stages 0, 1, 2
    assert 0 in profile
    assert 1 in profile
    assert 2 in profile
    
    # Validate strictly decreasing pattern: stage 0 > stage 1 > stage 2
    assert profile[0] > profile[1]
    assert profile[1] > profile[2]
