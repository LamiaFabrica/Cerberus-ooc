import os

files=[
   r"/c/McMaker Projects/Projects/Cerberus - Main/code/src/cerberus_command_executor.cpp",
   r"/c/McMaker Projects/Projects/Cerberus - Main/code/include/hq/intel_npu_telemetry.hpp",
   r"/c/McMaker Projects/Projects/Cerberus - Main/code/src/intel_npu_telemetry.cpp",
   r"/c/McMaker Projects/Projects/Cerberus - Main/code/include/hq/npu_backend_unified.hpp",
   r"/c/McMaker Projects/Projects/Cerberus - Main/code/src/npu_backend_unified.cpp"
]
forbidden=['simulate','minimal','for now','skeleton','heuristic','placeholder','stub']
for path in files:
    try:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            lines=f.read().splitlines()
        for i,line in enumerate(lines,1):
            lower=line.lower()
            for w in forbidden:
                if w in lower:
                    print(f"VIOLATION {path}:{i}:{line.strip()}")
    except Exception as e:
        print(path,'ERROR',e)
