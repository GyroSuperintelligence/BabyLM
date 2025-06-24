# GyroSI Cryptography: Traceable, Navigation-Driven Encryption

## Overview

The `ext_Cryptographer` extension implements a traceable, navigation-aware stream cipher for GyroSI. It provides per-user symmetric encryption for data at rest, fusing user-keyed security with entropy derived from navigation history. This approach ensures both strong cryptographic guarantees and deep integration with the system's operational semantics.

---

## Design Goals & Guarantees

- **Traceability:** The cryptographic state is fully determined by the user key and navigation history, ensuring reproducible encryption/decryption.
- **RAM Safety:** Gene/Genome data remain plaintext in memory; encryption is only applied at storage boundaries.
- **Navigation Integrity:** Navigation mechanics are never affected by cryptography.
- **Minimal Footprint:** The extension uses only 5 bytes of state (2B counter, 2B gyration evolution, 1B evolution counter).
- **Auditability:** All state transitions are traceable and can be analyzed for security and debugging.

---

## Core Concepts

### User Key
- Must be 16–32 bytes.
- Used as the cryptographic root for all operations.

### Gyration Evolution
- A 16-bit value that acts as a salt for the keystream.
- **Traceable Initialization:** The gyration evolution is derived from the first two bytes of `BLAKE2s(user_key)`, ensuring that any instance with the same key starts from the same gyration evolution.
- Evolves over time based on navigation events and encryption activity.

### Counter
- 16-bit stream counter, incremented for each 16-byte block encrypted.
- Ensures keystream uniqueness for each block.

### Evolution Counter
- 8-bit value tracking the number of gyration evolution.

---

## Encryption Process

1. **Chunking:** Data is processed in 16-byte blocks.
2. **Gyration Evolution:** Before each block (except the first), the gyration evolution is evolved using navigation history and a spin transformation.
3. **Keystream Generation:** For each block, a keystream is generated using BLAKE2s keyed with the user key, block index, and current gyration evolution.
4. **XOR Encryption:** The plaintext block is XORed with the keystream to produce ciphertext.
5. **Counter Update:** The counter is incremented by the number of blocks processed.

---

## Decryption Process

- **Symmetric:** Decryption mirrors encryption, using the same chunking, gyration evolution, and keystream generation.
- **Stateless Support:** Decryption always starts from counter 0 and the initial gyration evolution, matching the behavior of a fresh instance. This ensures round-trip correctness even after storage or transfer.
- **State Restoration:** After decryption, the cryptographic state (counter, gyration evolution, evolution counter) is restored to its original values.

---

## Navigation-Driven Entropy & Gyration Evolution

- **Navigation Events:** The extension tracks up to 16 recent navigation events.
- **Gyration Evolution:**
    - Every 16 bytes encrypted, or every 8 navigation events, the gyration evolution evolves.
    - If fewer than 8 navigation events are present, the history is padded with zeros.
    - Entropy is computed by combining navigation events with bitwise operations.
    - A spin transformation (bit rotation) is applied, and the result is XORed with the entropy to produce the new gyration evolution.
- **Result:** This mechanism ensures that the cryptographic state is sensitive to both user key and operational history, making attacks more difficult and providing a unique fusion of cryptography and system behavior.

---

## Diagnostic & Analysis Tools

- **Crypto Analysis:** The extension provides methods to inspect the current gyration evolution, evolution counter, statistics (bytes encrypted/decrypted, evolution count, entropy quality), and recent gyration evolution values.
- **Entropy Quality:** The distribution of gyration evolution bits is analyzed to estimate entropy quality, helping to detect weak or repetitive states.
- **Period Detection:** The extension can detect if the gyration evolution is cycling with a fixed period, which could indicate a problem.

---

## Example Usage

```python
from src.extensions.ext_cryptographer import ext_Cryptographer

key = b"my_super_secret_key_1234"
crypto = ext_Cryptographer(key)

plaintext = b"Sensitive data here..."
ciphertext = crypto.encrypt(plaintext)
# ... store or transmit ciphertext ...

# Decrypt (can use a new instance with the same key)
crypto2 = ext_Cryptographer(key)
decrypted = crypto2.decrypt(ciphertext)
assert decrypted == plaintext
```

---

## Security Notes

- **Key Management:** Security depends on the secrecy and quality of the user key.
- **Traceability:** All cryptographic operations are reproducible given the same key and navigation history.
- **No RAM Encryption:** Data is only encrypted at storage boundaries, never in RAM.
- **Navigation Sensitivity:** Navigation events directly influence cryptographic state, providing additional entropy and system integration.

---

## Summary

The GyroSI cryptography extension (`ext_Cryptographer`) provides a robust, traceable, and navigation-aware encryption layer. Its design ensures both strong security and deep integration with the system's operational semantics, making it suitable for advanced, auditable, and lineage-sensitive applications. 