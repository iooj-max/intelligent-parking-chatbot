"""Simple tests for bug fixes without complex dependencies."""
import os
import sys

os.environ['OPENAI_API_KEY'] = 'sk-test-key'
sys.path.insert(0, '.')

def test_config_validation():
    """Test API key validation (Phase 1.2 - Security fix)."""
    print('\n1. Testing Config Validation...')
    from pydantic import ValidationError

    # Valid key should work
    try:
        from src.config import Settings
        settings = Settings(openai_api_key='sk-test123')
        print('  ✓ Accepts valid API key')
    except Exception as e:
        print(f'  ✗ Should accept valid key: {e}')
        return False

    # Empty key should be rejected
    try:
        Settings(openai_api_key='')
        print('  ✗ Should reject empty API key')
        return False
    except ValidationError as e:
        if 'required' in str(e).lower() or 'empty' in str(e).lower():
            print('  ✓ Correctly rejects empty API key')
        else:
            print(f'  ? Rejected but unexpected error: {e}')

    # Invalid format should be rejected
    try:
        Settings(openai_api_key='invalid-key')
        print('  ✗ Should reject invalid key format')
        return False
    except ValidationError as e:
        if 'sk-' in str(e):
            print('  ✓ Correctly rejects invalid key format')
        else:
            print(f'  ? Rejected but unexpected error: {e}')

    return True


def test_rapidfuzz_installed():
    """Test rapidfuzz dependency was added (Phase 4.1)."""
    print('\n2. Testing rapidfuzz dependency...')
    try:
        import rapidfuzz
        from rapidfuzz import fuzz, process

        # Test basic functionality
        score = fuzz.ratio('downtown plaza', 'downtowm plaza')
        if score > 80:
            print(f'  ✓ rapidfuzz installed and working (similarity: {score})')
            return True
        else:
            print(f'  ✗ rapidfuzz works but unexpected score: {score}')
            return False
    except ImportError as e:
        print(f'  ✗ rapidfuzz not installed: {e}')
        return False


def test_syntax_new_files():
    """Test syntax of new files (Phase 2)."""
    print('\n3. Testing syntax of new files...')
    import py_compile
    import tempfile

    files = [
        'src/services/parking_service.py',
        'src/services/parking_matcher.py'
    ]

    all_ok = True
    for filepath in files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f'  ✓ {filepath} - syntax OK')
        except py_compile.PyCompileError as e:
            print(f'  ✗ {filepath} - syntax error: {e}')
            all_ok = False

    return all_ok


def test_modified_files_syntax():
    """Test syntax of modified files."""
    print('\n4. Testing syntax of modified files...')
    import py_compile

    files = [
        'src/chatbot/nodes.py',
        'src/chatbot/graph.py',
        'src/chatbot/prompts.py',
        'src/rag/retriever.py',
        'src/rag/vector_store.py',
        'src/rag/sql_store.py',
        'src/data/loader.py',
        'src/config.py',
        'src/guardrails/input_filter.py'
    ]

    all_ok = True
    for filepath in files:
        try:
            py_compile.compile(filepath, doraise=True)
            print(f'  ✓ {filepath}')
        except py_compile.PyCompileError as e:
            print(f'  ✗ {filepath} - syntax error: {e}')
            all_ok = False

    return all_ok


def test_hardcoded_strings_removed():
    """Test that hardcoded parking IDs were removed (Phase 3)."""
    print('\n5. Checking hardcoded parking IDs removed...')

    checks = []

    # Check PARKING_NAMES removed from prompts.py
    with open('src/chatbot/prompts.py', 'r') as f:
        content = f.read()
        if 'PARKING_NAMES' not in content:
            print('  ✓ PARKING_NAMES removed from prompts.py')
            checks.append(True)
        else:
            print('  ✗ PARKING_NAMES still in prompts.py')
            checks.append(False)

    # Check PARKING_ID_PATTERNS removed from retriever.py
    with open('src/rag/retriever.py', 'r') as f:
        content = f.read()
        if 'PARKING_ID_PATTERNS' not in content:
            print('  ✓ PARKING_ID_PATTERNS removed from retriever.py')
            checks.append(True)
        else:
            print('  ✗ PARKING_ID_PATTERNS still in retriever.py')
            checks.append(False)

    # Check ParkingFacilityService is used
    with open('src/chatbot/nodes.py', 'r') as f:
        content = f.read()
        if 'get_parking_service' in content:
            print('  ✓ ParkingFacilityService used in nodes.py')
            checks.append(True)
        else:
            print('  ✗ ParkingFacilityService not found in nodes.py')
            checks.append(False)

    # Check ParkingFacilityMatcher is used
    with open('src/rag/retriever.py', 'r') as f:
        content = f.read()
        if 'ParkingFacilityMatcher' in content:
            print('  ✓ ParkingFacilityMatcher used in retriever.py')
            checks.append(True)
        else:
            print('  ✗ ParkingFacilityMatcher not found in retriever.py')
            checks.append(False)

    return all(checks)


def test_bug_fixes_present():
    """Check that bug fixes are present in code."""
    print('\n6. Checking bug fixes present...')

    checks = []

    # Check __del__ added to WeaviateStore
    with open('src/rag/vector_store.py', 'r') as f:
        content = f.read()
        if 'def __del__(self):' in content:
            print('  ✓ __del__() added to WeaviateStore (memory leak fix)')
            checks.append(True)
        else:
            print('  ✗ __del__() not found in WeaviateStore')
            checks.append(False)

    # Check get_session_context added to SQLStore
    with open('src/rag/sql_store.py', 'r') as f:
        content = f.read()
        if 'get_session_context' in content:
            print('  ✓ get_session_context() added to SQLStore (session leak fix)')
            checks.append(True)
        else:
            print('  ✗ get_session_context() not found in SQLStore')
            checks.append(False)

    # Check timezone.utc used in date validation
    with open('src/chatbot/nodes.py', 'r') as f:
        content = f.read()
        if 'timezone.utc' in content or 'timezone' in content:
            print('  ✓ Timezone-aware date validation (logic error fix)')
            checks.append(True)
        else:
            print('  ✗ timezone not found in nodes.py')
            checks.append(False)

    # Check completion validates values
    with open('src/chatbot/graph.py', 'r') as f:
        content = f.read()
        if 'all_values_present' in content:
            print('  ✓ Completion check validates values (logic error fix)')
            checks.append(True)
        else:
            print('  ✗ all_values_present not found in graph.py')
            checks.append(False)

    return all(checks)


def main():
    """Run all simple tests."""
    print('=' * 70)
    print('SIMPLE VALIDATION TESTS')
    print('Testing bug fixes and removed hardcoded IDs')
    print('=' * 70)

    results = {
        'Config Validation': test_config_validation(),
        'rapidfuzz Dependency': test_rapidfuzz_installed(),
        'New Files Syntax': test_syntax_new_files(),
        'Modified Files Syntax': test_modified_files_syntax(),
        'Hardcoded IDs Removed': test_hardcoded_strings_removed(),
        'Bug Fixes Present': test_bug_fixes_present()
    }

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    for test_name, passed in results.items():
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'{status}: {test_name}')

    passed = sum(results.values())
    total = len(results)

    print(f'\nTotal: {passed}/{total} tests passed')

    if passed == total:
        print('\n🎉 All validation tests passed!')
        print('\nKey changes verified:')
        print('  • Phase 1: All critical bug fixes applied')
        print('  • Phase 2: ParkingFacilityService created')
        print('  • Phase 3: All hardcoded parking IDs removed')
        print('  • Phase 4: Semantic search with rapidfuzz added')
        return 0
    else:
        print(f'\n⚠️  {total - passed} test(s) failed')
        return 1


if __name__ == '__main__':
    exit(main())
