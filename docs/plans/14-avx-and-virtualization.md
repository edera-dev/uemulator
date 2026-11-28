# AVX and nested virtualization

**Status:** partially implemented. AVX, AVX2, FMA, F16C, AVX-512 and SVM are implemented (03). VMX below is a design only and is not advertised.

## AVX, AVX2, FMA, F16C

Done: see 03, "AVX, AVX2, FMA, F16C" and "AVX-512" (including AVX512_BF16, SHA-NI and ADX). VP2INTERSECT (`cpu/avx512_misc.rs`) and AVX-VNNI are done too. Not done: AVX512_FP16, the Xeon Phi sets (PF, ER).

## SVM (AMD-V) for nested guests

Done, for guests in any CPU mode: see 03, "SVM (AMD-V)", including the limits. AVIC and x2AVIC (`cpu/svm_avic.rs`) and decode assists are done, and a real hypervisor runs on it: `hack/kvm/run.sh` has Linux's kvm_amd start vCPUs in every CPU mode. What remains: a guest that itself executes `vmrun` (a hypervisor inside the guest hypervisor's guest), which is intercepted like any other VMRUN and #UD without that intercept (`cpu/svm.rs` module docs).

## VMX (Intel VT-x)

Not a fit for the AMD identity (`GenuineIntel` would also change Linux's model-specific quirks and mitigation tables, see 03 "Speculation control"). VMCS fields via `vmread`/`vmwrite`, EPT, VPID, the capability MSRs 0x480-0x491: a larger surface than SVM with no benefit for this project unless an Intel-only guest hypervisor must run. Defer until SVM works and then only if needed.

## Other items not done

- ERMS/FSRM: done (03, "ERMS and FSRM").
- PKU, CET shadow stacks and LA57: done (03; `cpu/pku.rs`, `cpu/cet.rs`, `cpu/la57_tests.rs`; shadow stacks in the JIT: 06). XSAVEC/XSAVES and x2APIC/APIC are done too, see 03.
