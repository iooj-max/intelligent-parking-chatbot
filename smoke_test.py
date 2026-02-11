"""Quick smoke tests for bug fixes and new features."""
import os
import sys

# Set up test environment
os.environ['OPENAI_API_KEY'] = 'sk-test-key-for-testing'
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'parking'
os.environ['POSTGRES_USER'] = 'parking'
os.environ['POSTGRES_PASSWORD'] = 'parking'

sys.path.insert(0, '.')

def test_config_validation():
    """Test API key validation (Security fix)."""
    from src.config import Settings
    from pydantic import ValidationError

    # Valid key should work
    try:
        settings = Settings(openai_api_key='sk-test123')
        print('✓ Config accepts valid API key')
    except Exception as e:
        print(f'✗ Config should accept valid key: {e}')
        return False

    # Empty key should be rejected
    try:
        settings = Settings(openai_api_key='')
        print('✗ Config should reject empty API key')
        return False
    except ValidationError:
        print('✓ Config correctly rejects empty API key')

    # Invalid format should be rejected
    try:
        settings = Settings(openai_api_key='invalid-key')
        print('✗ Config should reject invalid format')
        return False
    except ValidationError:
        print('✓ Config correctly rejects invalid key format')

    return True


def test_parking_matcher():
    """Test ParkingFacilityMatcher fuzzy matching."""
    from src.services.parking_matcher import ParkingFacilityMatcher
    from src.rag.sql_store import ParkingFacility

    # Create test facilities
    facilities = [
        ParkingFacility(
            parking_id='downtown_plaza',
            name='Downtown Plaza Parking',
            address='123 Main Street',
            city='City',
            total_spaces=100
        ),
        ParkingFacility(
            parking_id='airport_parking',
            name='Airport Long-Term Parking',
            address='4500 Airport Blvd',
            city='City',
            total_spaces=200
        )
    ]

    matcher = ParkingFacilityMatcher(threshold=0.6)

    # Test exact match
    matches = matcher.match_facility('Downtown Plaza Parking', facilities)
    if matches and matches[0]['parking_id'] == 'downtown_plaza':
        print('✓ Matcher handles exact match')
    else:
        print('✗ Matcher failed exact match')
        return False

    # Test fuzzy match with typo
    matches = matcher.match_facility('downtowm plaza', facilities)  # typo
    if matches and matches[0]['parking_id'] == 'downtown_plaza':
        print('✓ Matcher handles typo matching')
    else:
        print('✗ Matcher failed typo matching')
        return False

    # Test partial match
    matches = matcher.match_facility('airport', facilities)
    if matches and matches[0]['parking_id'] == 'airport_parking':
        print('✓ Matcher handles partial match')
    else:
        print('✗ Matcher failed partial match')
        return False

    return True


def test_csv_sanitization():
    """Test CSV sanitization (Security fix)."""
    from src.data.loader import DataLoader

    loader = DataLoader(verbose=False)

    # Test formula injection prevention
    tests = [
        ('=1+1', "'=1+1"),
        ('+cmd', "'+cmd"),
        ('-formula', "'-formula"),
        ('@import', "'@import"),
        ('normal text', 'normal text'),
        (123, 123),  # Non-string should pass through
    ]

    all_passed = True
    for input_val, expected in tests:
        result = loader._sanitize_csv_value(input_val)
        if result == expected:
            print(f'✓ Sanitize: {repr(input_val)} -> {repr(result)}')
        else:
            print(f'✗ Sanitize failed: {repr(input_val)} -> {repr(result)} (expected {repr(expected)})')
            all_passed = False

    return all_passed


def test_dsn_redaction():
    """Test DSN password redaction (Security fix)."""
    from src.rag.sql_store import SQLStore

    store = SQLStore(dsn='postgresql://user:secret_password@localhost:5432/db')
    redacted = store._redact_password(store.dsn)

    if 'secret_password' not in redacted and '****' in redacted:
        print(f'✓ DSN password redacted: {redacted}')
        return True
    else:
        print(f'✗ DSN password not redacted: {redacted}')
        return False


def main():
    """Run all smoke tests."""
    print('=' * 60)
    print('SMOKE TESTS FOR BUG FIXES AND NEW FEATURES')
    print('=' * 60)
    print()

    results = {}

    print('1. Testing Config Validation (Security Fix)...')
    results['config'] = test_config_validation()
    print()

    print('2. Testing ParkingFacilityMatcher (Semantic Search)...')
    results['matcher'] = test_parking_matcher()
    print()

    print('3. Testing CSV Sanitization (Security Fix)...')
    results['csv'] = test_csv_sanitization()
    print()

    print('4. Testing DSN Redaction (Security Fix)...')
    results['dsn'] = test_dsn_redaction()
    print()

    print('=' * 60)
    print('RESULTS:')
    print('=' * 60)
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = '✓ PASS' if result else '✗ FAIL'
        print(f'{status}: {test_name}')

    print()
    print(f'Total: {passed}/{total} tests passed')

    if passed == total:
        print('\n🎉 All smoke tests passed!')
        return 0
    else:
        print(f'\n⚠️  {total - passed} test(s) failed')
        return 1


if __name__ == '__main__':
    exit(main())
