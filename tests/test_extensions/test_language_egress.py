from src.extensions.ext_LanguageEgress import ext_LanguageEgress


def test_language_generation():
    """Test basic language generation from navigation"""
    egress = ext_LanguageEgress()

    # Simulate 48 navigation events (one complete cycle)
    nav_events = [0x12, 0x34, 0x56, 0x78] * 12  # 48 events

    captured_output = []

    # Mock the emit function
    def mock_emit(text):
        captured_output.append(text)

    egress._emit_text = mock_emit

    # Process events
    for event in nav_events:
        egress.process_navigation_event(event)

    # Should have processed one complete cycle
    assert len(egress.cycle_buffer) == 0  # Buffer cleared after processing
    assert len(egress.text_buffer) > 0  # Some text generated


def test_sentence_detection():
    """Test sentence boundary detection"""
    egress = ext_LanguageEgress()

    # Add text ending with period
    egress.text_buffer = bytearray(b"Hello world.")

    captured = []
    egress._emit_text = lambda t: captured.append(t)

    egress._emit_complete_sentences()

    assert len(captured) == 1
    assert captured[0] == "Hello world."
    assert len(egress.text_buffer) == 0  # Buffer cleared
