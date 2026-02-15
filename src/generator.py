import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.prosody import validate_line

# ===== MODEL PATHS =====
BASE_MODEL = "abhinand/tamil-llama-7b-base-v0.1"
LORA_PATH = "/content/drive/MyDrive/tamil_poetry_lora_final" 

print(" Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="offload",     
    offload_state_dict=True        
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("Model ready!\n")


# ===== THEME KEYWORDS =====
THEME_KEYWORDS = {
    "அறம்": ["அறம்", "நன்மை", "தர்மம்", "நீதி", "ஒழுக்கம்", "நேர்மை", "தயை", "இரக்கம்", "நல்ல"],
    "காதல்": ["காதல்", "அன்பு", "காதலன்", "காதலி", "காமம்", "பிரிவு", "சந்திப்பு", "நேசம்", "காதல"],
    "இயற்கை": ["மரம்", "மலர்", "காடு", "மலை", "நதி", "கடல்", "வானம்", "நிலா", "சூரியன்", "பூ", "கிளை"],
    "வாழ்க்கை": ["வாழ்க்கை", "வாழ்வு", "பிறப்பு", "இறப்பு", "உழைப்பு", "பயணம்", "காலம்"],
    "அன்பு": ["அன்பு", "அன்னை", "தாய்", "தந்தை", "குழந்தை", "உறவு", "பாசம்", "நேசம்"],
    "தேசபக்தி": ["தேசம்", "நாடு", "தமிழ்", "விடுதலை", "சுதந்திரம்", "வீரம்", "போராட்டம்"],
    "பெண்ணுரிமை": ["பெண்", "மகள்", "தாய்", "சமத்துவம்", "உரிமை", "விடுதலை", "சுதந்திரம்"],
}


# ===== BALANCED QUALITY CHECKS =====
def has_theme_relevance(line, theme):
    """
    Check if line contains theme-related keywords.
    """
    if theme not in THEME_KEYWORDS:
        return True
    
    keywords = THEME_KEYWORDS[theme]
    
    for keyword in keywords:
        if keyword in line:
            return True
    
    return False


def is_complete_and_meaningful(line):
    """
    BALANCED check: Line should be complete and somewhat meaningful.
    Not too strict - focuses on critical issues only.
    """
    
    words = line.strip().split()
    if len(words) < 2:
        return False
    
    last_word = words[-1]
    
    # CRITICAL: Reject clearly incomplete endings
    clearly_incomplete = ["அதனை", "இதனை", "என்றால்", "தந்தால்"]
    if last_word in clearly_incomplete:
        return False
    
    # CRITICAL: Reject if ends with pulli (consonant cluster)
    if last_word.endswith("்"):
        return False
    
    # Otherwise accept - don't be too picky
    return True


def is_appropriate_length(line, poet="ஒளவையார்"):
    """
    Check word count - PRACTICAL ranges.
    """
    word_count = len(line.split())
    
    if poet == "ஒளவையார்":
        # Auvaiyar: 3-6 words (short maxims)
        return 3 <= word_count <= 6
    elif poet == "பாரதிதாசன்":
        # Bharathidasan: 4-7 words (REDUCED - was 5-10)
        # More similar to Auvaiyar for better quality
        return 4 <= word_count <= 7
    else:
        return 3 <= word_count <= 8


def has_basic_grammar(line):
    """
    BASIC grammar check - only catch serious problems.
    """
    
    # Reject excessive spaces
    if line.count("  ") > 3:
        return False
    
    # Reject weird punctuation
    if line.count(".") > 3 or line.count(",") > 4:
        return False
    
    # That's it - don't be too strict on grammar
    return True


def has_narrative_markers(line):
    """
    Check for obvious prose markers.
    """
    narrative_words = ["என்றான்", "என்றாள்", "என்றார்", "கூறினார்", "சொன்னான்", "சொன்னாள்"]
    return any(word in line for word in narrative_words)


# ================= PROMPT BUILDER =================
def build_prompt(theme, poet="ஒளவையார்", previous_lines=None):
    """
    Simple, clear prompt.
    """
    
    if poet == "ஒளவையார்":
        instruction = f"{poet} நடைபோல் {theme} கருவில் ஒரு குறள் போன்ற கவிதை வரி எழுதுக."
    elif poet == "பாரதிதாசன்":
        instruction = f"{poet} நடைபோல் {theme} கருவில் ஒரு உணர்ச்சி மிகுந்த கவிதை வரி எழுதுக."
    else:
        instruction = f"{poet} நடைபோல் {theme} கருவில் ஒரு கவிதை வரி எழுதுக."
    
    # Add context if available
    if previous_lines and len(previous_lines) > 0:
        context = "\n".join(previous_lines)
        instruction += f"\n\nமுந்தைய வரிகள்:\n{context}"
    
    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    return prompt


# ================= LINE GENERATION =================
def generate_line(theme, poet="ஒளவையார்", previous_lines=None):
    """
    Generate with BALANCED parameters.
    """
    
    prompt = build_prompt(theme, poet, previous_lines)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # BALANCED parameters - not too strict, not too loose
    if poet == "ஒளவையார்":
        max_tokens = 12        # Reasonable for 3-6 words
        temp = 0.65            # Moderate creativity
    elif poet == "பாரதிதாசன்":
        max_tokens = 14        # REDUCED (was 20) for 4-7 words
        temp = 0.7
    else:
        max_tokens = 15
        temp = 0.65

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = text.replace(prompt, "").strip()
    
    # Clean up
    if "\n" in generated:
        generated = generated.split("\n")[0].strip()
    
    if "###" in generated:
        generated = generated.split("###")[0].strip()

    banned_phrases = ["கவிஞன்", "எழுது", "வரிகள்", "instruction", "assistant", "கருவில்"]
    for phrase in banned_phrases:
        if phrase in generated:
            return ""

    return generated.strip()


# ================= BALANCED POEM GENERATION =================
def generate_poem(theme, poet="ஒளவையார்", num_lines=4, max_attempts=12):
    """
    Generate with BALANCED quality checks.
    Focus on: complete, short, theme-relevant.
    """
    
    poem_lines = []
    
    print(f"\n{'='*60}")
    print(f"🎯 Generating {num_lines}-line poem (BALANCED MODE)")
    print(f"Poet: {poet} | Theme: {theme}")
    print(f"{'='*60}\n")

    for i in range(num_lines):
        print(f"--- Line {i+1}/{num_lines} ---")

        accepted_line = None
        
        for attempt in range(max_attempts):
            line = generate_line(theme, poet, previous_lines=poem_lines)
            
            print(f"  [{attempt+1:2d}] '{line}'", end="")
            
            # ============ BALANCED QUALITY CHECKS ============
            # Focus on the essentials only!
            
            # Check 1: Not empty
            if not line:
                print("Empty")
                continue
            
            # Check 2: Reasonable length (3-6 words for Auvaiyar)
            if not is_appropriate_length(line, poet):
                word_count = len(line.split())
                print(f" {word_count}w")
                continue
            
            # Check 3: Not duplicate
            if line in poem_lines:
                print(" Dup")
                continue
            
            # Check 4: Complete and meaningful (not too strict)
            if not is_complete_and_meaningful(line):
                print(" Incomplete")
                continue
            
            # Check 5: Basic grammar (minimal check)
            if not has_basic_grammar(line):
                print("Grammar")
                continue
            
            # Check 6: No prose markers
            if has_narrative_markers(line):
                print("Prose")
                continue
            
            # Check 7: Theme relevance (only first 4 attempts)
            if attempt < 4:
                if not has_theme_relevance(line, theme):
                    print("Theme")
                    continue
            
            # Check 8: Prosody (only first 6 attempts)
            if attempt < 6:
                if not validate_line(line):
                    print("Meter")
                    continue
            
            # === LINE ACCEPTED ===
            print("OK")
            accepted_line = line
            break
        
        # Fallback if needed
        if not accepted_line:
            print(f"\n Using best available after {max_attempts} attempts")
            accepted_line = line if line else f"[வரி {i+1}]"
        
        poem_lines.append(accepted_line)
        print()

    # Display final poem
    print(f"\n{'='*60}")
    print("📝 FINAL POEM:")
    print(f"{'='*60}")
    for i, line in enumerate(poem_lines, 1):
        print(f"{i}. {line}")
    print(f"{'='*60}\n")
    
    return poem_lines


# ================= BATCH GENERATION =================
def generate_multiple_poems(theme, poet="ஒளவையார்", num_poems=3, num_lines=4):
    """
    Generate multiple poems and pick the best.
    """
    all_poems = []
    
    for i in range(num_poems):
        print(f"\n{'#'*60}")
        print(f"POEM {i+1}/{num_poems}")
        print(f"{'#'*60}")
        
        poem = generate_poem(theme, poet, num_lines)
        all_poems.append(poem)
    
    print(f"\n{'='*60}")
    print("ALL POEMS:")
    print(f"{'='*60}\n")
    
    for i, poem in enumerate(all_poems, 1):
        print(f"--- POEM {i} ---")
        for line in poem:
            print(line)
        print()
    
    return all_poems


# ================= USAGE =================
if __name__ == "__main__":
    
    # Single poem
    poem = generate_poem(
        theme="அறம்",
        poet="ஒளவையார்",
        num_lines=4
    )
    
    # Or generate multiple options:
    # poems = generate_multiple_poems(
    #     theme="அறம்",
    #     poet="ஒளவையார்",
    #     num_poems=3,
    #     num_lines=4
    # )
