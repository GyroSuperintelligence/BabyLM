Development Guide
===============

This guide outlines the development practices and requirements for contributing to GyroSI.

Project Structure
--------------

The project follows a strict structural organization:

```
gyro_si/
├── audit/        # Runtime trace data (G5)
├── benchmarks/   # Performance tests
├── data/         # Versioned data (DVC)
├── docs/         # Project documentation (Sphinx)
├── gyro_si/      # Core source code (G1-G6)
├── patterns/     # Canonical data for G1
├── scripts/      # Utility scripts
├── tests/        # Correctness tests
└── ...           # Project config files
```

Development Requirements
---------------------

1. **Structural Compliance**
   - All code must adhere to the recursive tensor-based architecture
   - No ad-hoc additions or modifications to the core structure
   - Maintain strict stage-to-folder isomorphism

2. **Code Quality**
   - Type hints required for all functions
   - Comprehensive docstrings following Google style
   - Unit tests for all new functionality
   - Performance benchmarks for critical paths

3. **Memory Management**
   - All memory operations must be lineage-tagged
   - Checksums required for all data structures
   - Clear separation of memory types (G1-G5)

4. **Testing Requirements**
   - Unit tests for all tensor operations
   - Integration tests for system interactions
   - Performance regression tests
   - Memory leak detection

5. **Documentation**
   - API documentation for all public interfaces
   - Architecture documentation for system components
   - Development guides for common tasks
   - Performance optimization guides

Development Workflow
-----------------

1. **Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install dependencies
   pip install -e ".[dev]"
   ```

2. **Development**
   - Create feature branch from main
   - Implement changes following structural requirements
   - Add tests and documentation
   - Run test suite and fix issues

3. **Testing**
   ```bash
   # Run tests
   pytest
   
   # Run with coverage
   pytest --cov=gyro_si
   
   # Run performance tests
   pytest benchmarks/
   ```

4. **Code Quality**
   ```bash
   # Format code
   black .
   isort .
   
   # Type checking
   mypy .
   
   # Linting
   flake8
   ```

5. **Documentation**
   ```bash
   # Build documentation
   cd docs
   make html
   ```

6. **Review**
   - Submit pull request
   - Address review comments
   - Ensure CI passes
   - Merge to main

Best Practices
------------

1. **Tensor Operations**
   - Use explicit tensor forms
   - Maintain discrete value constraints
   - Follow canonical sequence

2. **Memory Management**
   - Tag all memory operations
   - Maintain checksums
   - Follow memory type separation

3. **Error Handling**
   - Use quantization error for observation
   - Propagate algedonic signals
   - Maintain structural alignment

4. **Performance**
   - Profile critical paths
   - Optimize tensor operations
   - Monitor memory usage

5. **Security**
   - Validate all inputs
   - Maintain structural integrity
   - Follow security guidelines 