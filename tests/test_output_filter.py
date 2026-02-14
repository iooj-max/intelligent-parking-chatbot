from src.guardrails.output_filter import OutputFilter


def test_phone_values_are_captured_and_masked():
    output_filter = OutputFilter()

    result = output_filter.filter_response(
        "Call me at +1-555-123-4567 or (555) 123-4567."
    )

    assert result["is_safe"] is True
    assert result["severity"] == "low"
    assert result["filtered_response"] == "Call me at [PHONE] or [PHONE]."
    assert result["pii_found"] == [
        {"type": "phone", "value": "+1-555-123-4567"},
        {"type": "phone", "value": "(555) 123-4567"},
    ]


def test_high_severity_pii_blocks_response():
    output_filter = OutputFilter()

    result = output_filter.filter_response(
        "SSN: 123-45-6789 and card: 4111 1111 1111 1111"
    )

    assert result["is_safe"] is False
    assert result["severity"] == "high"
    assert "cannot provide that information" in result["filtered_response"]
