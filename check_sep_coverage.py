#!/usr/bin/env python3

from baby.policies import OrbitStore
from pathlib import Path

store = OrbitStore("memories/public/knowledge/knowledge.bin", use_mmap=True)
count, unique_states = 0, set()

for (s, t), e in store.iter_entries():
    if t == 102 and e.get("direction", 0) == 0:
        count += 1
        unique_states.add(s)

print("SEP entries:", count, "unique pre-states:", len(unique_states))
print("Total entries:", sum(1 for _ in store.iter_entries()))
print("SEP coverage ratio:", count / sum(1 for _ in store.iter_entries()) if sum(1 for _ in store.iter_entries()) > 0 else 0) 