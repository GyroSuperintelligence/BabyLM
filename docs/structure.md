├── CHANGELOG.md
├── LICENSE
├── Makefile
├── README.md
├── config
│   ├── extensions.config.yaml
│   └── gyro_config.yaml
├── core
├── gyro_cli.py
├── gyro_tools
│   ├── __init__.py
│   ├── build_operator_matrix.py
│   ├── cli
│   │   ├── __init__.py
│   │   ├── commands
│   │   │   ├── __init__.py
│   │   │   ├── curriculum.py
│   │   │   ├── integrity.py
│   │   │   ├── knowledge.py
│   │   │   ├── session.py
│   │   │   └── system.py
│   │   ├── config.py
│   │   ├── interactive.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── display.py
│   │       └── prompts.py
│   ├── gyro_cli.py
│   ├── gyro_curriculum_manager.py
│   ├── gyro_integrity_check.py
│   ├── gyro_knowledge_manager.py
│   └── gyro_session_manager.py
├── install.sh
├── pyproject.toml
├── pyrightconfig.json
├── requirements.txt
├── run.py
├── scripts
│   ├── benchmark.py
│   ├── build_app.py
│   ├── build_release.py
│   ├── convert_svg_to_png.py
│   ├── dev.py
│   ├── dev_hot_reload.py
│   ├── run_tests.py
│   └── setup_dev.py
├── src
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── alignment_nav.py
│   │   ├── extension_manager.py
│   │   ├── gyro_api.py
│   │   ├── gyro_core.py
│   │   ├── gyro_errors.py
│   │   ├── gyro_tag_parser.py
│   │   └── log_decoder.py
│   ├── extensions
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ext_api_gateway.py
│   │   ├── ext_bloom_filter.py
│   │   ├── ext_coset_knowledge.py
│   │   ├── ext_cryptographer.py
│   │   ├── ext_error_handler.py
│   │   ├── ext_event_classifier.py
│   │   ├── ext_fork_manager.py
│   │   ├── ext_language_egress.py
│   │   ├── ext_multi_resolution.py
│   │   ├── ext_navigation_helper.py
│   │   ├── ext_performance_tracker.py
│   │   ├── ext_phase_controller.py
│   │   ├── ext_resonance_processor.py
│   │   ├── ext_spin_piv.py
│   │   ├── ext_state_helper.py
│   │   ├── ext_storage_manager.py
│   │   └── ext_system_monitor.py
│   ├── gyro_api.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core
│   │   ├── __init__.py
│   │   ├── test_alignment_nav.py
│   │   ├── test_gyro_core.py
│   │   ├── test_integration.py
│   │   ├── test_mechanical_engine.py
│   │   ├── test_rle_compression.py
│   │   └── test_tag_parser.py
│   ├── test_extensions
│   │   ├── __init__.py
│   │   ├── test_bloom_filter.py
│   │   ├── test_coset_knowledge.py
│   │   ├── test_language_egress.py
│   │   ├── test_multi_resolution.py
│   │   └── test_spin_piv.py
│   ├── test_frontend
│   │   ├── __init__.py
│   │   ├── test_components.py
│   │   └── test_gyro_app.py
│   ├── test_integration_complete.py
│   ├── test_knowledge
│   │   ├── __init__.py
│   │   ├── test_export_import.py
│   │   ├── test_knowledge_forking.py
│   │   └── test_session_linking.py
│   └── test_tools
│       ├── __init__.py
│       ├── test_cli.py
│       └── test_knowledge_manager.py
