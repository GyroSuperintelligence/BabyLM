# GyroSI Baby LM Storage and Encryption Model

## 1. Canonical Genome Pack Format

- **Location:** `agency/g1_information/<shard>/genome_<start_idx:012x>.dat`
- **Pack Size:** 64 MB (configurable, default 64,000,000 bytes)
- **Header:** 128 bytes, self-indexing, no SHA-256 anchor, no per-cycle index
- **Struct Packing:** All headers use '<4sBBHI' (little-endian, no spaces)
- **Atomicity:** GenomePack appends are protected by fcntl.flock for thread/process safety
- **Agent-private files:** Always .json.enc, 128B header, GyroCrypt
- **File Naming:** `genome_<start_idx:012x>.dat` (deterministic, agent-agnostic)
- **Cycles:** 24 bytes each, packed, cleartext (not encrypted)
- **Naming:** Deterministic, agent-agnostic, lexicographically sortable by starting cycle index
- **Header Fields:**

| Offset | Size | Field                | Notes                                              |
|--------|------|----------------------|----------------------------------------------------|
| 0      | 4    | magic = b'GYRO'      | sanity check                                       |
| 4      | 1    | version = 2          | format version                                     |
| 5      | 1    | flags = 0            | always 0 for genome packs                          |
| 6      | 2    | header_size = 128    | allows future extension                            |
| 8      | 4    | start_cycle_index    | first cycle number stored                          |
| 12     | 96   | gene_stateless_snapshot      | output of get_gene_stateless_snapshot()              |
| 108    | 12   | salt                 | random per pack, for keystream separation (unused) |
| 120    | 8    | reserved (zero)      | pad to 128                                         |

- **Cycles:** Each cycle is 24 bytes (48 op-pairs, 2 per byte)
- **Auditability:** Genome packs are always cleartext, public, and agent-agnostic

## 2. Agent-Private File Encryption

- **Files:**
  - `agents/<uuid>/g3_information/format.json.enc`
  - `agents/<uuid>/g5_information/session.json.enc`
- **Header:** 128 bytes, same structure as genome packs but with different magic
  - `b'GYR3'` for format
  - `b'GYR5'` for session
- **Payload:** UTF-8 JSON, chunked into 24-byte blocks, XORed with GyroCrypt keystream, padded to 24B boundary
- **Keystream:** Deterministic, derived from 96B snapshot, 12B salt, and agent key
- **Salt:** 12 random bytes per file, ensures keystream uniqueness even for identical snapshots/keys
- **Extension:** `.json.enc` for all encrypted agent-private files
- **Best Practice:** Only agent with the correct key can decrypt; files are not readable by others

## 3. Keystream and Salt Rationale

- **Keystream:** `make_keystream(snapshot, salt, key)`
  - Deterministic, physics-native, no external entropy
  - Salt ensures per-file uniqueness, prevents keystream reuse
- **Salt:**
  - 12 bytes, random per file
  - Optionally encodes provenance (e.g., agent/session/timestamp)
  - Cryptographic hygiene: prevents replay and keystream collision

## 4. File Layout Examples

```
agency/g1_information/00/genome_000000000000.dat
agents/00/abcdef1234567890/g3_information/format.json.enc
agents/00/abcdef1234567890/g5_information/session.json.enc
```

## 5. Backward Compatibility

- **Legacy files:**
  - Plaintext `format.json` and `session.json` are still readable (fallback)
  - Migration tool can convert legacy files to encrypted format
- **Genome packs:** Only new format is supported for writing; loader can distinguish by magic

## 6. What Is and Isn't Encrypted

| File/Folder                                      | Encrypted? | Notes                                 |
|--------------------------------------------------|------------|---------------------------------------|
| `agency/g1_information/*.dat`                    | No         | Genome is public, agent-agnostic      |
| `agents/<uuid>/g3_information/format.json.enc`| Yes        | Agent-private, GyroCrypt-encrypted    |
| `agents/<uuid>/g5_information/session.json.enc`   | Yes        | Agent-private, GyroCrypt-encrypted    |
| `agency/g4_information/*.json`                   | No         | Global dictionaries, public           |
| `agency/g2_information/g2_information.dat`        | No         | Epigenome, public                     |

## 7. Auditability and Privacy

- **Genome is always clear:** Anyone can hash, audit, and replay the genome
- **Agent privacy is preserved:** Only the agent's key can decrypt their format and session
- **Performance:** Encryption overhead is negligible (only two small files per agent)

## 8. Test and Migration Expectations

- **Tests:**
  - All tests expect `.json.enc` for agent-private files
  - Use EncryptedFile helpers for round-trip verification
- **Migration:**
  - Legacy plaintext files are supported for reading
  - Migration tool can encrypt existing agent files in place

## Concurrent Write Safety

- All genome pack appends are atomic and thread-safe via `fcntl.flock`.
- Tests validate that concurrent writes do not interleave or corrupt data.

## Header Validation in Tests

- Tests check for 128B header, correct magic, and struct layout in all packs and encrypted files.

For further details, see the implementation in `s4_intelligence/g1_intelligence_in.py` and the test cases in `scripts/tests/`. 