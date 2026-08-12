import pytest
from ntt_engine import NTTEngine
from verification_engine import ScheduleVerifier

def test_karatsuba_memory_accesses():
    """Verify that Karatsuba mode produces more accesses than Schoolbook mode,
    and has exactly 512 intermediate scratch accesses (2 writes + 2 reads per instance) for N=256.
    """
    n = 256
    modulus = 3329
    engine = NTTEngine(n=n, modulus=modulus)
    
    # Input polynomials
    poly_a = list(range(1, n + 1))
    poly_b = [x * 2 for x in poly_a]
    
    # Run full pipeline in both modes
    result_sb = engine.run_full_mlkem_pipeline(poly_a, poly_b, mode="schoolbook")
    result_kt = engine.run_full_mlkem_pipeline(poly_a, poly_b, mode="karatsuba")
    
    # Extract steps in the base-mul stage (stage 7 for N=256)
    sb_base_mul_steps = [s for s in result_sb.execution_log if s.stage_number == engine.log_n - 1]
    kt_base_mul_steps = [s for s in result_kt.execution_log if s.stage_number == engine.log_n - 1]
    
    # Count total memory accesses (reads + writes) in the base-mul stage
    sb_total_accesses = sum(len(s.inputs) + len(s.outputs) for s in sb_base_mul_steps)
    kt_total_accesses = sum(len(s.inputs) + len(s.outputs) for s in kt_base_mul_steps)
    
    # 1. Confirm Karatsuba produces MORE total memory accesses
    assert kt_total_accesses > sb_total_accesses
    assert sb_total_accesses == 128 * (4 + 2)  # Schoolbook: 128 instances of 4 reads + 2 writes = 768
    assert kt_total_accesses == 2048           # Karatsuba: 128 * 8 (final) + 256 * 3 (write) + 256 * 1 (read) = 2048
    
    # 2. Confirm the intermediate steps alone produce exactly 512 accesses to the scratch region
    # (2 writes + 2 reads per base-mul instance, i.e. 4 * 128 = 512 total)
    intermediate_steps = [s for s in kt_base_mul_steps if s.op_type == "KARATSUBA_INTERMEDIATE"]
    scratch_writes = sum(len([addr for addr in s.outputs if addr >= 2 * n]) for s in intermediate_steps)
    scratch_reads = sum(len([addr for addr in s.inputs if addr >= 2 * n]) for s in intermediate_steps)
    
    assert len(intermediate_steps) == 512
    assert scratch_writes == 256
    assert scratch_reads == 256
    assert scratch_writes + scratch_reads == 512
    
    # 3. Confirm that the final BASE_MUL_KARATSUBA nodes' inputs include the scratch addresses
    final_kt_steps = [s for s in kt_base_mul_steps if s.op_type == "BASE_MUL_KARATSUBA"]
    assert len(final_kt_steps) == 128
    for s in final_kt_steps:
        assert len(s.inputs) == 6
        # The last two inputs must be the scratch addresses allocated for this instance
        idx0 = s.inputs[0]
        i = idx0 // 2
        scratch_addr_1 = 2 * n + 2 * i
        scratch_addr_2 = 2 * n + 2 * i + 1
        assert s.inputs[4] == scratch_addr_1
        assert s.inputs[5] == scratch_addr_2

    # 4. Verify that the DAG structure is valid, acyclic, and passes the verifier checks
    verifier = ScheduleVerifier.from_ntt_result(result_kt)
    dep_check = verifier.verify_dependencies()
    assert dep_check.passed, f"Dependency verification failed: {dep_check.violations}"
