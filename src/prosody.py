"""Prosody validation module for Tamil poetry.

NOTE: This is a simplified prosody checker. For production use, 
consider using proper Tamil NLP libraries like:
- tamil library (pip install tamil)
- Indic NLP library

Current limitations:
- Basic syllable splitting (doesn't handle all edge cases)
- Simplified NER/NIRAI classification
- Generic Venpa rules (not line-position specific)
"""

import re

# Tamil vowel letters (standalone)
SHORT_VOWELS = ["அ", "இ", "உ", "எ", "ஒ"]
LONG_VOWELS = ["ஆ", "ஈ", "ஊ", "ஏ", "ஐ", "ஓ", "ஔ"]

# Tamil vowel diacritics (attached to consonants)
SHORT_VOWEL_SIGNS = ["\u0BBE", "\u0BBF", "\u0BC1", "\u0BC6", "\u0BCA"]  # ா ி ு ெ ொ
LONG_VOWEL_SIGNS = ["\u0BC0", "\u0BC2", "\u0BC7", "\u0BC8", "\u0BCB", "\u0BCC"]  # ீ ூ ே ை ோ ௌ

# Consonants
CONSONANTS = ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", 
              "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"]

# Pulli (virama) - makes consonant pure
PULLI = "\u0BCD"  # ்

# Tamil unicode range
TAMIL_START = 0x0B80
TAMIL_END = 0x0BFF


# --- BASIC TAMIL CHARACTER CHECK ---
def is_tamil_char(ch):
    """Check if character is in Tamil Unicode range."""
    return TAMIL_START <= ord(ch) <= TAMIL_END


# --- IMPROVED SYLLABLE SPLITTING ---
def split_syllables(word):
    """
    Split a Tamil word into syllables.
    
    NOTE: This is a simplified implementation. For better accuracy,
    use dedicated Tamil NLP libraries.
    
    Basic rules:
    - Standalone vowel = one syllable
    - Consonant + vowel mark = one syllable  
    - Consonant + pulli = part of next syllable or end
    """
    
    if not word:
        return []
    
    syllables = []
    chars = list(word)
    i = 0
    
    while i < len(chars):
        if not is_tamil_char(chars[i]):
            i += 1
            continue
            
        syllable = chars[i]
        
        # If standalone vowel, that's the syllable
        if chars[i] in SHORT_VOWELS + LONG_VOWELS:
            syllables.append(syllable)
            i += 1
            continue
        
        # If consonant, check what follows
        if chars[i] in CONSONANTS:
            i += 1
            
            # Collect any following vowel signs or pulli
            while i < len(chars) and is_tamil_char(chars[i]):
                # Check if it's a vowel sign or pulli
                if ord(chars[i]) >= 0x0BBE and ord(chars[i]) <= 0x0BCD:
                    syllable += chars[i]
                    i += 1
                    
                    # If we hit pulli, syllable might continue
                    if chars[i-1] == PULLI:
                        # Check if next char is consonant (double consonant)
                        if i < len(chars) and chars[i] in CONSONANTS:
                            # This is a conjunct, include next consonant
                            syllable += chars[i]
                            i += 1
                    break
                else:
                    break
            
            syllables.append(syllable)
        else:
            i += 1
    
    return [s for s in syllables if s]  # Filter empty strings


# --- CLASSIFY ASAI (NER / NIRAI) ---
def classify_asai(syllables):
    """
    Classify syllables as NER (long) or NIRAI (short).
    
    Simplified rule:
    - NER: Contains long vowel (ஆ, ஈ, ஊ, ஏ, ஐ, ஓ, ஔ or their signs)
    - NIRAI: Contains short vowel
    
    NOTE: Real prosody also considers consonant clusters, position, etc.
    """
    asai_pattern = []
    
    for syl in syllables:
        # Check for long vowel signs
        has_long = any(sign in syl for sign in LONG_VOWEL_SIGNS)
        # Check for standalone long vowels
        has_long = has_long or any(v in syl for v in LONG_VOWELS)
        
        if has_long:
            asai_pattern.append("NER")
        else:
            asai_pattern.append("NIRAI")
    
    return asai_pattern


# --- COUNT ASAI ---
def count_asai(line):
    """
    Count the asai (metrical units) in a line.
    
    Returns:
        tuple: (asai_count, asai_pattern)
    """
    words = line.split()
    all_syllables = []
    
    for word in words:
        syllables = split_syllables(word)
        all_syllables.extend(syllables)
    
    asai_pattern = classify_asai(all_syllables)
    return len(asai_pattern), asai_pattern


# --- VENPA METER RULES ---
def venpa_rule_check(asai_count, line_number=None):
    """
    Check if asai count matches Venpa meter rules.
    
    NOTE: This is simplified. Real Venpa has specific rules for each line:
    - Lines 1 & 2: Usually 4-5 asai
    - Line 3: Usually 4-5 asai
    - Line 4: Usually 3-4 asai
    
    For now, we use a broad range since we don't always know line position.
    """
    
    # If we know the line number, apply stricter rules
    if line_number is not None:
        if line_number in [1, 2, 3]:
            return 3 <= asai_count <= 8
        elif line_number == 4:
            return 2 <= asai_count <= 7
    
    # Generic Venpa range (when line position unknown)
    # RELAXED: Allow 2-10 asai for research purposes
    return 2 <= asai_count <= 10


# --- MAIN VALIDATION FUNCTION (SCORE-BASED) ---
def validate_line(line, line_number=None):
    """
    Validate line and return a prosody score between 0.0 and 1.0.
    
    Args:
        line: The Tamil text line to validate
        line_number: Optional line position in poem (1-4 for Venpa)
    
    Returns:
        float: Prosody score from 0.0 (invalid) to 1.0 (perfect)
    """
    line = line.strip()
    score = 0.0
    
    # Check 1: Not empty (20%)
    if not line:
        return 0.0
    score += 0.2
    
    # Check 2: Minimum word count (20%)
    words = line.split()
    if len(words) < 2:
        return score
    score += 0.2
    
    # Check 3: Tamil content present (20%)
    if not any(is_tamil_char(c) for c in line):
        return score
    score += 0.2
    
    # Check 4: Asai count analysis (40%)
    asai_count, pattern = count_asai(line)
    
    # Too short
    if asai_count < 2:
        return score
    
    # Too long
    if asai_count > 15:
        return score
    
    # Check Venpa meter compliance
    if venpa_rule_check(asai_count, line_number):
        # Perfect meter match
        score += 0.4
    else:
        # Partial credit based on how close to ideal range
        ideal_min = 3
        ideal_max = 8
        
        if asai_count < ideal_min:
            # Too short - give partial credit
            distance = ideal_min - asai_count
            partial = max(0, 0.4 - (distance * 0.1))
            score += partial
        elif asai_count > ideal_max:
            # Too long - give partial credit
            distance = asai_count - ideal_max
            partial = max(0, 0.4 - (distance * 0.05))
            score += partial
    
    return min(score, 1.0)  # Cap at 1.0


def validate_line_boolean(line, line_number=None):
    """
    Legacy boolean validation for backward compatibility.
    Returns True if prosody score >= 0.6
    """
    score = validate_line(line, line_number)
    return score >= 0.6


# --- DEBUGGING / ANALYSIS FUNCTIONS ---
def analyze_line(line):
    """
    Detailed analysis of a line's prosody.
    Useful for debugging.
    """
    words = line.split()
    print(f"\nAnalyzing: {line}")
    print(f"Words: {len(words)}")
    
    all_syllables = []
    for word in words:
        syllables = split_syllables(word)
        all_syllables.extend(syllables)
        print(f"  {word} → {syllables}")
    
    asai_count, pattern = count_asai(line)
    print(f"\nTotal syllables: {len(all_syllables)}")
    print(f"Asai count: {asai_count}")
    print(f"Pattern: {' '.join(pattern)}")
    print(f"Venpa valid: {venpa_rule_check(asai_count)}")
    
    return asai_count, pattern


# --- TESTING ---
if __name__ == "__main__":
    # Test with some known Tamil poetry lines
    test_lines = [
        "அறம் செய விரும்பு",
        "ஆறுவது சினம்",
        "இயல்பு அலாதன செய்யேல்",
        "அத்தைனைத் தவிர மறத்தோர் வேறேனும் உண்டா என்றான்"  # Bad example (too long)
    ]
    
    print("="*60)
    print("Testing prosody validation")
    print("="*60)
    
    for line in test_lines:
        print(f"\nLine: {line}")
        is_valid = validate_line(line)
        print(f"Valid: {is_valid}")
        
        if not is_valid:
            analyze_line(line)