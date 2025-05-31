#!/usr/bin/env python3
"""
tests/core/test_lambda_foundation.py
Test Suite for Lambda Calculus Foundation

Validates mathematical correctness of function composition
and DOP compliance enforcement.

Author: OBINexus Engineering Team / Nnamdi Okpala
Phase: 3.1 - Core Foundation Implementation
"""

import pytest
from datetime import datetime, timedelta
from pyics.core import Lambda, ImmutableEvent, shift_event_time

def test_lambda_identity():
    """Test identity function behavior"""
    test_value = "test"
    assert Lambda.identity(test_value) == test_value

def test_lambda_composition():
    """Test function composition correctness"""
    def add_one(x): return x + 1
    def multiply_two(x): return x * 2
    
    composed = Lambda.compose(multiply_two, add_one)
    assert composed(3) == 8  # (3 + 1) * 2 = 8

def test_lambda_pipe():
    """Test left-to-right piping"""
    def add_one(x): return x + 1
    def multiply_two(x): return x * 2
    
    piped = Lambda.pipe(add_one, multiply_two)
    assert piped(3) == 8  # (3 + 1) * 2 = 8

def test_immutable_event_creation():
    """Test immutable event structure"""
    event = ImmutableEvent(
        uid="test-001",
        summary="Test Event",
        start_time=datetime(2024, 12, 30, 9, 0),
        duration=timedelta(hours=1)
    )
    
    assert event.uid == "test-001"
    assert event.summary == "Test Event"

def test_transformation_immutability():
    """Test that transformations preserve immutability"""
    original_event = ImmutableEvent(
        uid="test-002",
        summary="Original Event",
        start_time=datetime(2024, 12, 30, 9, 0),
        duration=timedelta(hours=1)
    )
    
    transform = shift_event_time(timedelta(hours=1))
    modified_event = transform(original_event)
    
    # Original should be unchanged
    assert original_event.start_time == datetime(2024, 12, 30, 9, 0)
    # Modified should have shifted time
    assert modified_event.start_time == datetime(2024, 12, 30, 10, 0)
    # Objects should be different instances
    assert original_event is not modified_event

if __name__ == "__main__":
    pytest.main([__file__])
