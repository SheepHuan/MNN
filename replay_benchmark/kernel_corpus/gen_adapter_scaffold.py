#!/usr/bin/env python3
"""Generate OpAdapter scaffolding for all MNN OpenCL baked kernels.

Scans operators.json for all opencl/mnn entries, reads the first __kernel
signature from each baked .cl file, and generates a .hpp/.cpp pair with
adapter classes that set up the correct argument layout.

Generated adapters use identity_fp32 validator and fill buffers with
deterministic test data. They can be refined per-operator later.
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_kernel_signature(source):
    """Extract first __kernel void name(...) signature."""
    m = re.search(r'__kernel\s+(?:__attribute__\s*\([^)]*\)\s+)?void\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', source, re.DOTALL)
    if not m:
        return None, None
    name = m.group(1)
    params = m.group(2).strip()
    return name, params


def classify_param(param, param_idx):
    """Classify a kernel parameter into an AdaptedArg kind."""
    param = param.strip()
    # Remove const qualifiers
    param = re.sub(r'__private\s+const\s+', '', param)
    param = re.sub(r'__private\s+', '', param)
    param = re.sub(r'const\s+', '', param)
    
    # Size dim params (from GLOBAL_SIZE macros)
    if re.match(r'int\s+global_size_dim\d', param) or re.match(r'int\s+global_dim\d', param):
        dim = int(re.search(r'global_size_dim(\d)|global_dim(\d)', param).group(1) or re.search(r'global_size_dim(\d)|global_dim(\d)', param).group(2))
        return f'sizeConst({dim})'
    
    # Buffer params
    if '__global' in param:
        return 'buffer'
    
    # int2
    if 'int2' in param:
        return 'int2'
    
    # int4
    if 'int4' in param:
        return 'int4'
    
    # Scalar int
    if 'int' in param and 'int2' not in param and 'int4' not in param:
        return 'scalarInt'
    
    # Scalar float
    if 'float' in param:
        return 'scalarFloat'
    
    return 'unknown'


def main():
    corpus_root = Path('replay_benchmark/kernel_corpus')
    ops = json.load(open(corpus_root / 'operators.json'))
    
    opencl_ops = [o for o in ops['operators'] if o['backend'] == 'opencl']
    
    # Group by op_type
    by_op = defaultdict(list)
    for o in opencl_ops:
        by_op[o['op_type']].append(o)
    
    print(f"Found {len(opencl_ops)} opencl operators in {len(by_op)} op_types")
    
    for op_type, ops_list in sorted(by_op.items()):
        for o in ops_list:
            cl_path = corpus_root / o['file']
            if not cl_path.exists():
                print(f"SKIP (missing): {o['variant']}")
                continue
            source = cl_path.read_text(encoding='utf-8', errors='replace')
            name, params = parse_kernel_signature(source)
            if not name:
                print(f"SKIP (no kernel): {o['variant']}")
                continue
            
            # Parse params (split by comma at top level)
            param_list = []
            depth = 0
            current = ''
            for c in params:
                if c in '(<':
                    depth += 1
                elif c in ')>':
                    depth -= 1
                if c == ',' and depth == 0:
                    param_list.append(current.strip())
                    current = ''
                else:
                    current += c
            if current.strip():
                param_list.append(current.strip())
            
            # Classify each param
            classifications = []
            buf_idx = 0
            for i, p in enumerate(param_list):
                cls = classify_param(p, i)
                if cls == 'buffer':
                    classifications.append(f'buffer({buf_idx})')
                    buf_idx += 1
                else:
                    classifications.append(cls)
            
            print(f"OP {o['op_type']:25} TAG {o['tag']:6} VARIANT {o['variant']:35} ENTRY {name:30} ARGS {classifications}")


if __name__ == '__main__':
    main()
