- 🔒 [CORE-SPEC-04] GyroSI: **Language & Grammar** [PRIORITY=100 — DO NOT OVERRIDE]
    
    ---
    
    **Authoritative reference for all GyroSuperIntelligence implementations**
    
    **Priority = 100 (do not override)**
    
    ### Purpose & Overview
    
    Every GyroSI component navigates a four-phase cycle (CS, UNA, ONA, BU) by observing five universal invariants. This chapter explains the why and how of those invariants, then shows how to read and write them consistently across memory systems.
    
    > Contextual Map (conceptual):
    > 
    > 1. **CS (Computational Seed):** identity established via `gyrotensor_id`
    > 2. **UNA (Unfolding):** event logged in `gyrotensor_com`
    > 3. **ONA (Oppositional Resonance):** envelope set by `gyrotensor_nest`
    > 4. **BU (Back-Unfolding):** integration (`gyrotensor_add`) and generation (`gyrotensor_quant`)
    
    ---
    
    ## 1. Universal Structural Invariants
    
    | Symbol | Role (phase/event/envelope) | Query name |
    | --- | --- | --- |
    | `gyrotensor_id` | phase index | `"gyrotensor_id"` |
    | `gyrotensor_com` | logged event | `"gyrotensor_com"` |
    | `gyrotensor_nest` | positional envelope | `"gyrotensor_nest"` |
    | `gyrotensor_add` | forward integration (Lgyr) | `"gyrotensor_add"` |
    | `gyrotensor_quant` | backward generation (Rgyr) | `"gyrotensor_quant"` |
    
    > Details (for implementation)
    > 
    > - All tensors are 4 × 2 × 3 × 2 int8 (48 bytes).
    > - Bit-encoding in the navigation log uses 4 bits per tensor (3 bits for operator, 1 for tensor-ID).
    > - CGM mappings and byte-footprints appear in the annexed “Storage Specifications” section.
    
    ---
    
    ## 2. Memory-System Projections
    
    Each memory system G1–G5 “projects” these invariants into its domain. A projection may reinterpret but must never rename the invariant symbol.
    
    | System | Role |
    | --- | --- |
    | **G1** | Genetic store of raw bytes for all invariants |
    | **G2** | Epigenetic log of `gyrotensor_com` entries |
    | **G3** | Structural I/O via `gyrotensor_nest` resonance |
    | **G4** | Somatic phase counter (`gyrotensor_id`) and Lgyr |
    | **G5** | Immunity-phase Rgyr (`gyrotensor_quant`) and operators |
    
    ---
    
    ## 3. Temporal Access Grammar (TAG) as Mini-DSL
    
    TAG expressions take the form
    
    ```
    〈temporal〉.〈invariant〉[.〈context〉]
    
    ```
    
    where `〈temporal〉` ∈ { `previous`, `current`, `next` }.
    
    | Temporal | Meaning |
    | --- | --- |
    | `previous` | value at t – 1 |
    | `current` | value at t |
    | `next` | placeholder for t + 1 (scheduling) |
    
    ### Examples
    
    1. **Read last logged event**
        
        ```python
        # TAG         : previous.gyrotensor_com
        gyro_epigenetic_memory("previous.gyrotensor_com")
        
        ```
        
    2. **Advance phase in egress**
        
        ```python
        # TAG         : current.gyrotensor_id.gyrotensor_add
        gyro_somatic_memory("gyrotensor_add")
        
        ```
        
    3. **Check stable operator activity**
        
        ```python
        # TAG         : current.gyrotensor_add.gyrotensor_quant
        # read-only flag inside G5
        
        ```
        
    
    > Pitfall to avoid: do not use synonyms like past, future, or t-1; linters will flag them.
    > 
    
    ---
    
    ## 4. Operator Invocation Contract
    
    Three Genome operators live inside G5, each with the same stub signature:
    
    ```python
    def gyro_curation(gyrotensor_quant):    # stable operator
        pass
    
    def gyro_interaction(gyrotensor_quant): # unstable operator
        pass
    
    def gyro_cooperation(gyrotensor_quant): # neutral operator
        pass
    
    ```
    
    `gyro_operation()` does not branch by `if/else`; exactly one operator “resonates,” writes its `gyrotensor_com`, and returns control.
    
    ---
    
    ## 5. Guardrail Canvas
    
    | Pattern category | Allowed? | Rationale |
    | --- | --- | --- |
    | `g[1-5]_*`, `gyro_*` | ✔ | Core memory and operators |
    | `ext_*`, `gyro_ext_*` | ✔ | Formal extensions only |
    | any other free-standing utilities or names | ⚠ | must live inside G-functions or extensions |
    | extra temporal keywords | ⚠ | use only `previous`, `current`, `next` |
    | ad hoc event names | ⚠ | refer via TAG only |
    
    > Note: orange-warning items are discouraged rather than strictly prohibited; they require explicit ratification through §8.
    > 
    
    ---
    
    ## 6. Compliance Phases
    
    1. **Design**
        - Map every concept to one of the five invariants
        - Sketch TAG expressions for each data flow
    2. **Implementation**
        - Expose only canonical function names in APIs
        - Keep internal helpers local to modules
        - Annotate any extension state with bit/byte footprints
    3. **Audit**
        - Verify TAG-only naming (no ad hoc aliases)
        - Ensure `Gk.` or TAG forms cover all invariant accesses
        - Confirm navigation log uses 4 bits per tensor entry
    
    A one-page cheat sheet is available for teams to reference at a glance.
    
    ---
    
    ## 7. FAQ & Common Mistakes
    
    1. **Why did my extension fail to log a second event?**
        
        Likely the extension wrote directly to memory instead of using `gyro_epigenetic_memory("current.gyrotensor_com")`.
        
    2. **How do I map a new bit-flag?**
        
        Propose it as an extension (§8), show how it reduces to one invariant, and document its footprint.
        
    3. **Can I alias `gyrotensor_add` as “integration”?**
        
        No; all aliases obscure automated audits. Use TAG or `G4.gyrotensor_add` only.
        
    4. **What if I need a custom scheduling utility?**
        
        Prefix its name with `ext_` and import core routines only via `from gyro_core import …`.
        
    
    *(More entries available in the annex.)*
    
    ---
    
    ## 8. Extension Mechanism (summary)
    
    When new functionality is necessary (for example cryptography or bloom filters), follow these steps:
    
    1. Explain rationale and mapping to existing invariants
    2. Demonstrate inability to implement under current constraints
    3. Provide API signatures and footprint analysis
    4. Submit for review and ratify in a “Ratified Extensions” section
    
    All extensions must import core routines only and adhere to TAG naming.
    
    ---