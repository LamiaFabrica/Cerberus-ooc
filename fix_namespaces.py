import re

fpath = r'code/src/david_propup_engine.cpp'
with open(fpath, 'r', encoding='utf-8') as f:
    src = f.read()

replacements = [
    (r'\bTieredMemoryConfig\b', 'hq::TieredMemoryConfig'),
    (r'\bTieredMemoryManager\b', 'hq::TieredMemoryManager'),
    (r'\bMemoryTier\b', 'hq::MemoryTier'),
    (r'\bTierAllocation\b', 'hq::TierAllocation'),
    (r'\bTierHandle\b', 'hq::TierHandle'),
    (r'\bTierError\b', 'hq::TierError'),
    (r'\bTierStats\b', 'hq::TierStats'),
    (r'\bKernelGraph\b', 'hq::npu::KernelGraph'),
    (r'\bKernelNode\b', 'hq::npu::KernelNode'),
    (r'\bCompiledKernel\b', 'hq::npu::CompiledKernel'),
    (r'\bTensorDesc\b', 'hq::npu::TensorDesc'),
    (r'\bCpuFallbackBackend\b', 'hq::npu::CpuFallbackBackend'),
    (r'\bTargetConfig\b', 'hq::npu::TargetConfig'),
    (r'\bQuantMethod\b', 'hq::npu::QuantMethod'),
    (r'\bQuantGranularity\b', 'hq::npu::QuantGranularity'),
    (r'\bCpuPostProcessor\b', 'hq::npu::CpuPostProcessor'),
    (r'\bIntelNpuTelemetry\b', 'hq::npu::IntelNpuTelemetry'),
    (r'\bNpuBackendFactory\b', 'hq::npu::NpuBackendFactory'),
    (r'\bCerberusGraph\b', 'hq::cerberus::CerberusGraph'),
    (r'\bDecisionConfig\b', 'hq::cerberus::DecisionConfig'),
    (r'\bDecisionEngine\b', 'hq::cerberus::DecisionEngine'),
    (r'\bCerberusExecutionCoordinator\b', 'hq::CerberusExecutionCoordinator'),
    (r'\bStagingConfig\b', 'hq::StagingConfig'),
    (r'\bEmbeddingStagingManager\b', 'hq::EmbeddingStagingManager'),
    (r'\bScopedTierAlloc\b', 'hq::ScopedTierAlloc'),
]

for pat, repl in replacements:
    src = re.sub(r'(?<!hq::)(?<!hq::npu::)(?<!hq::cerberus::)(?<!::)' + re.escape(pat.split('::')[-1]) + r'', repl, src)

with open(fpath, 'w', encoding='utf-8') as f:
    f.write(src)

print('wrote', fpath, 'size', len(src))
