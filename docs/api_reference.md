# API Reference Structure

This document lists all modules, classes, and functions in the BabyLM codebase (src/ and gyro_tools/).

---

## src/main.py
- (No classes)
- (No functions)

---

## src/gyro_api.py
- Functions:
  - initialize_session
  - shutdown_session
  - list_active_sessions
  - get_session_info
  - process_byte
  - process_byte_stream
  - process_text
  - process_file
  - export_knowledge
  - import_knowledge
  - fork_knowledge
  - link_session_to_knowledge
  - query_memory
  - get_navigation_history
  - validate_system_integrity
  - cleanup_inactive_sessions
  - _get_manager
  - _ensure_data_directories
  - get_language_output

---

## src/core/gyro_core.py
- Classes:
  - GyroEngine
    - __init__
    - _load_and_validate_harmonics
    - execute_cycle
    - _get_gene_constant
    - _compute_gene_checksum
    - load_phase
- Functions:
  - gyration_op

---

## src/core/extension_manager.py
- Classes:
  - ExtensionManager
    - __init__
    - _initialize_system_extensions
    - _initialize_application_extensions
    - _load_session_state
    - _validate_system_integrity
    - _cleanup_on_error
    - get_extension
    - get_session_id
    - get_knowledge_id
    - gyro_epigenetic_memory
    - gyro_structural_memory
    - gyro_somatic_memory
    - gyro_immunity_memory
    - gyro_operation
    - _notify_extensions
    - export_knowledge
    - import_knowledge
    - fork_knowledge
    - link_to_knowledge
    - get_system_health
    - shutdown
    - _get_event_tensor
    - _get_nest_tensor
    - _get_decoded_gene_state
    - _get_recent_events
    - _load_extension_states
    - _get_user_key
    - _handle_language_output
    - export_session
    - import_session

---

## src/core/alignment_nav.py
- Classes:
  - NavigationLog
    - __init__
    - append
    - _flush_identity_run
    - _append_raw_byte
    - iter_steps
    - step_count (property)
    - shutdown
    - load_from_disk
    - persist_to_disk
    - _prune
    - validate_gene_checksum

---

## src/core/gyro_errors.py
- Classes:
  - GyroError
  - GyroTagError
  - GyroPhaseError
  - GyroNoResonanceError
  - GyroIntegrityError
  - GyroImmutabilityError
  - GyroNavigationError
  - GyroForkError
  - GyroSessionError
  - GyroExtensionError
  - GyroStorageError
  - GyroConfigError

---

## src/core/gyro_tag_parser.py
- Functions:
  - validate_tag
  - parse_tag

---

## src/core/log_decoder.py
- Functions:
  - decode_log_stream
  - encode_identity_run

---

## src/core/__init__.py
- (No classes)
- (No functions)

---

## src/extensions/base.py
- Classes:
  - GyroExtension
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - ext_on_navigation_event
    - validate_footprint
    - get_pattern_filename
    - shutdown
    - persist_state
    - load_state

---

## src/extensions/ext_api_gateway.py
- Classes:
  - ext_APIGateway
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state

---

## src/extensions/ext_bloom_filter.py
- Classes:
  - ext_BloomFilter
    - __init__
    - ext_gyration_hash
    - ext_insert_pattern
    - ext_contains
    - get_saturation
    - get_false_positive_rate
    - get_theoretical_fpr
    - should_reset
    - reset
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - ext_on_navigation_event

---

## src/extensions/ext_coset_knowledge.py
- Classes:
  - ext_CosetKnowledge
    - __init__
    - ext_add_pattern
    - _find_coset_representative
    - _calculate_similarity
    - _update_compression_ratio
    - ext_get_coset_info
    - ext_get_semantic_groups
    - ext_reconstruct_pattern
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - ext_on_navigation_event

---

## src/extensions/ext_cryptographer.py
- Classes:
  - ext_Cryptographer
    - __init__
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - _generate_keystream
    - encrypt
    - decrypt
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - process_navigation_event
    - get_pattern_filename

---

## src/extensions/ext_error_handler.py
- Classes:
  - ext_ErrorHandler
    - __init__
    - handle_error
    - log_extension_error
    - _update_stats
    - _recover_phase_error
    - _recover_navigation_error
    - _recover_immutability_error
    - _recover_extension_error
    - _log_recovery
    - get_error_report
    - clear_error_state
    - has_critical_errors
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state

---

## src/extensions/ext_event_classifier.py
- Classes:
  - ext_EventClassifier
    - __init__
    - is_learning_event
    - classify_event
    - _track_event
    - _hash_event
    - get_event_statistics
    - get_recent_learning_events
    - register_learning_pattern
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state

---

## src/extensions/ext_fork_manager.py
- Classes:
  - ext_ForkManager
    - __init__
    - _update_immutability_status
    - is_current_knowledge_immutable
    - ensure_writable
    - fork
    - mark_immutable
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state

---

## src/extensions/ext_language_egress.py
- Classes:
  - ext_LanguageEgress
    - __init__
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - process_navigation_event
    - _process_complete_cycle
    - _navigation_to_text
    - _chunk_to_char
    - _emit_complete_sentences
    - _emit_text
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - get_pattern_filename
    - _extract_language_patterns
    - _get_character_mappings

---

## src/extensions/ext_multi_resolution.py
- Classes:
  - ext_MultiResolution
    - __init__
    - ext_on_navigation_event
    - _record_boundary
    - ext_get_boundary_analysis
    - ext_predict_next_boundary
    - ext_get_text_structure
    - process_navigation_event
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state

---

## src/extensions/ext_navigation_helper.py
- Classes:
  - ext_NavigationHelper
    - __init__
    - record_navigation
    - _get_boundary_type
    - _classify_operator
    - get_resonance_info
    - get_operator_info
    - _get_last_operators
    - predict_next_boundary
    - detect_navigation_patterns
    - get_cycle_progress
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - ext_on_navigation_event

---

## src/extensions/ext_performance_tracker.py
- Classes:
  - ext_PerformanceTracker
    - __init__
    - start_operation
    - end_operation
    - _get_memory_usage
    - _update_throughput
    - _record_alert
    - ext_get_operation_stats
    - ext_get_system_performance
    - ext_get_performance_report
    - ext_reset_metrics
    - get_extension_name
    - get_extension_version
    - get_footprint_bytes
    - get_learning_state
    - get_session_state
    - set_learning_state
    - set_session_state
    - _format_performance_summary
    - _calculate_efficiency_score

---

(Structure continues for all other files...)
