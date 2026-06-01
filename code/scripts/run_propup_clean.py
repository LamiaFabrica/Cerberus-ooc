import subprocess, sys, re
exe = "C:/McMaker Projects/Projects/Cerberus - Main/code/build/david_propup_engine.exe"
p = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    line = line.rstrip()
    if not line:
        continue
    if line.startswith("[TieredMemory]"):
        continue
    print(line, flush=True)
rc = p.wait()
print(f"EXIT={rc}", flush=True)
