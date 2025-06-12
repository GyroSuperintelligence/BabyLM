"""
Shared pytest configuration and fixtures for GyroSI test suite.

This module provides common fixtures and utilities used across all tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

@pytest.fixture
def mock_config():
    """
    Provides a mock configuration object for testing.

    Returns a mock with all config flags that can be easily toggled:
    - enable_bootstrap
    - enable_transactions
    - enable_recovery_beacons
    - enable_entropy_tracking
    - enable_crypto_evolution
    - lightweight_transactions
    - fast_entropy_hash
    """
    config = Mock()
    config.enable_bootstrap = True
    config.enable_transactions = True
    config.enable_recovery_beacons = True
    config.enable_entropy_tracking = False
    config.enable_crypto_evolution = False
    config.lightweight_transactions = True
    config.fast_entropy_hash = True
    return config

@pytest.fixture
def mock_router():
    """
    Provides a mock message router for testing communication.

    The router captures all sent messages and allows inspection
    of routing behavior without real async operations.
    """
    router = Mock()
    router.messages_sent = []
    router.subscribers = {}

    async def mock_send_message(message):
        router.messages_sent.append(message)

    def mock_subscribe(msg_type, handler):
        if msg_type not in router.subscribers:
            router.subscribers[msg_type] = []
        router.subscribers[msg_type].append(handler)

    router.send_message = AsyncMock(side_effect=mock_send_message)
    router.subscribe = Mock(side_effect=mock_subscribe)
    router.get_messages = AsyncMock(return_value=[])

    return router

@pytest.fixture
def sample_tensor_state():
    """
    Provides a sample tensor state dictionary for testing.

    Returns a dict with all required tensor fields at their
    initial values.
    """
    return {
        "tensor_id": 12345,
        "stage": "CS",
        "indptr": [0, 2, 4, 6],
        "indices": [0, 1, 0, 1, 0, 1],
        "data": [1, 1, 1, 1, 1, 1],
        "amplitude": 0.0,
        "cumulative_phase": 0.0,
        "chirality_phase": 0.0,
        "last_epsilon": 0.0,
        "cycle_index": 0
    }

@pytest.fixture
def sample_message():
    """
    Provides a sample message dictionary with all required fields.
    """
    return {
        "type": "test_message",
        "source": "G1",
        "destination": "G2",
        "cycle_index": 42,
        "tensor_context": {
            "cumulative_phase": 1.57,
            "chirality_phase": 1.57,
            "helical_position": 0.25,
            "spinor_cycle_count": 0
        },
        "payload": {"test": "data"},
        "timestamp": "2025-01-01T00:00:00Z"
    } 