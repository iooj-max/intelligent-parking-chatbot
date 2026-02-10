"""
Regex patterns for guardrail detection.

Contains compiled patterns for:
- Prompt injection detection
- Topic classification
- PII detection (emails, phones, SSNs, credit cards)
"""

import re

# PROMPT INJECTION PATTERNS
# Patterns that indicate attempts to override chatbot instructions
INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+(instructions?|prompts?)',
    r'forget\s+(what\s+)?you\s+know',
    r'system\s+prompt',
    r'act\s+as\s+a',
    r'you\s+are\s+now',
    r'override\s+(all\s+)?(instructions?|rules?)',
    r'bypass\s+(all\s+)?(security|validation)',
    r';?\s*drop\s+table',
    r';?\s*union\s+select',
    r'<script>',
    r'javascript:',
]

# TOPIC KEYWORDS FOR PARKING DOMAIN
# Keywords that indicate parking-related queries
PARKING_KEYWORDS = [
    'parking',
    'park',
    'space',
    'spot',
    'lot',
    'garage',
    'facility',
    'available',
    'availability',
    'hours',
    'open',
    'close',
    'price',
    'cost',
    'rate',
    'fee',
    'book',
    'reserve',
    'reservation',
    'location',
    'address',
    'downtown',
    'airport',
    'long-term',
    'short-term',
]

# OFF-TOPIC KEYWORDS
# Keywords that indicate queries outside parking domain
OFF_TOPIC_KEYWORDS = [
    'cryptocurrency',
    'bitcoin',
    'politics',
    'election',
    'religion',
    'medical',
    'health',
    'diagnosis',
    'legal',
    'lawyer',
    'financial advice',
    'investment',
    'stock',
    'trading',
]

# PII PATTERNS
# Regex patterns for detecting personally identifiable information

# Email pattern: matches standard email addresses
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Phone pattern: matches US phone numbers with various formats
# Examples: 555-1234, (555) 123-4567, +1-555-123-4567
# Also matches 7-digit local numbers: 555-1234
PHONE_PATTERN = r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{3}[-.\s]?\d{4}\b'

# SSN pattern: matches social security numbers (XXX-XX-XXXX)
# Uses word boundaries to avoid false positives with dates
SSN_PATTERN = r'\b\d{3}-\d{2}-\d{4}\b'

# Credit card pattern: matches 16-digit credit card numbers
# Handles spaces or dashes between groups
CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'

# COMPILED PATTERNS FOR PERFORMANCE
# Pre-compile all patterns to avoid repeated compilation overhead

# Injection patterns (case-insensitive)
INJECTION_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]

# PII patterns (case-sensitive for better accuracy)
EMAIL_REGEX = re.compile(EMAIL_PATTERN)
PHONE_REGEX = re.compile(PHONE_PATTERN)
SSN_REGEX = re.compile(SSN_PATTERN)
CARD_REGEX = re.compile(CREDIT_CARD_PATTERN)
