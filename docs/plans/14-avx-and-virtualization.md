# AVX and nested virtualization (not implemented)

Status: AVX, AVX2, FMA, F16C, AVX-512 and SVM are implemented (03). VMX below is a design only and is not advertised.

## AVX, AVX2, FMA, F16C

Done: see 03, "AVX, AVX2, FMA, F16C" and "AVX-512" (including AVX512_BF16, SHA-NI and ADX). Not done: AVX512_FP16/VP2INTERSECT, the Xeon Phi sets (PF, ER), AVX-VNNI.

## SVM (AMD-V) for nested guests

Done, for guests in any CPU mode: see 03, "SVM (AMD-V)", including the limits. What remains: a guest-side `vmrun` (nested virtualization), AVIC, decode assists, and above all a run against a real nested hypervisor.

## VMX (Intel VT-x)

Not a fit for the AMD identity (`GenuineIntel` would also change Linux's model-specific quirks and mitigation tables, see 03 "Speculation control"). VMCS fields via `vmread`/`vmwrite`, EPT, VPID, the capability MSRs 0x480-0x491: a larger surface than SVM with no benefit for this project unless an Intel-only guest hypervisor must run. Defer until SVM works and then only if needed.

## Other items not done

- ERMS/FSRM: done (03, "ERMS and FSRM").
- PKU, CET (shadow stacks; the kernel only needs `endbr64`, which is a NOP here), LA57. (XSAVEC/XSAVES and x2APIC/APIC are done, see 03.)
