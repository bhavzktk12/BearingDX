import os, re
from bk1 import RPM_BY_LOAD

def rpm_to_load(rpm: int) -> int:
    return min(RPM_BY_LOAD.items(), key=lambda kv: abs(kv[1]-int(rpm)))[0]

def infer_load_from_name(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"(\d{3,5})(?=\.mat$)", name)
    if m:
        return rpm_to_load(int(m.group(1)))
    m2 = re.search(r"normal_(\d)_\d+", name, re.I)
    return int(m2.group(1)) if m2 else 1
