"""
Bilingual Knowledge Distillation: Qwen3-Thinking (Teacher) → Phi3 (Student)
Transfer Khmer-English capabilities with mixed language tasks
Hardware: Core i9 + RTX 3090 24GB
"""

# ============================================================================
# STEP 1: INSTALLATION & SETUP
# ============================================================================
# Run these commands in terminal first:
"""
python3.10 -m venv distillation_env
source distillation_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets transformers torch pandas tqdm
"""

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
import json
from tqdm import tqdm
from transformers import TrainingArguments
from trl import SFTTrainer
import random
import os

# ============================================================================
# STEP 2: CONFIGURATION
# ============================================================================

TEACHER_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
STUDENT_MODEL = "unsloth/Phi-3-mini-4k-instruct"

class Config:
    # Teacher model (Qwen3-Thinking)
    TEACHER_MODEL = TEACHER_MODEL
    
    # Student model (Phi3)
    STUDENT_MODEL = STUDENT_MODEL
    
    # Training parameters
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    
    # LoRA parameters
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Output
    OUTPUT_DIR = "./phi3_khmer_english_distilled"
    
    # Data generation with language distribution
    NUM_SAMPLES = 1000
    KHMER_TO_ENGLISH_RATIO = 0.50  # 50%
    ENGLISH_TO_KHMER_RATIO = 0.30  # 30%
    KHMER_TO_KHMER_RATIO = 0.20    # 20%

config = Config()

# ============================================================================
# STEP 3: BILINGUAL PROMPT TEMPLATES
# ============================================================================

class BilingualPrompts:
    """Generate bilingual prompts with specific distributions"""
    
    # Khmer → English Translation Prompts (50%)
    KHMER_TO_ENGLISH = [
        "បកប្រែទៅជាភាសាអង់គ្លេស៖ បញ្ញាសិប្បនិម្មិតជាអ្វី?",
        "បកប្រែទៅជាភាសាអង់គ្លេស៖ ការរៀនម៉ាស៊ីនដំណើរការយ៉ាងណា?",
        "បកប្រែទៅជាភាសាអង់គ្លេស៖ តើអ្វីទៅជា neural network?",
        "Translate to English: ប្រទេសកម្ពុជាមានវប្បធម៌ចម្រុះ",
        "Translate to English: អង្គរវត្តគឺជាប្រាសាទធំបំផុតនៅលើពិភពលោក",
        "បកប្រែ៖ ភាសាខ្មែរមានប្រវត្តិយូរអង្វែង",
        "Translate to English: ការអភិវឌ្ឍន៍កម្មវិធីទាមទារជំនាញច្រើន",
        "បកប្រែ៖ ទិន្នន័យធំមានសារៈសំខាន់ក្នុងសម័យទំនើប",
        "Translate to English: ការសិក្សាអនឡាញកាន់តែពេញនិយម",
        "បកប្រែទៅអង់គ្លេស៖ បច្ចេកវិទ្យាផ្លាស់ប្តូរពិភពលោក",
    ]
    
    # English → Khmer Translation Prompts (30%)
    ENGLISH_TO_KHMER = [
        "Translate to Khmer: What is overfitting in machine learning?",
        "Translate to Khmer: Deep learning uses neural networks.",
        "បកប្រែទៅជាភាសាខ្មែរ៖ Artificial intelligence is transforming industries.",
        "Translate to Khmer: Natural language processing helps computers understand text.",
        "បកប្រែ៖ Data science combines statistics and programming.",
        "Translate to Khmer: Cloud computing provides scalable resources.",
        "បកប្រែទៅខ្មែរ៖ Software development requires careful planning.",
        "Translate to Khmer: Mobile applications are essential today.",
        "បកប្រែ៖ Cybersecurity protects digital assets.",
        "Translate to Khmer: Blockchain technology enables decentralization.",
    ]
    
    # Khmer → Khmer Tasks (20%)
    KHMER_TO_KHMER = [
        "សូមពន្យល់ពីបញ្ញាសិប្បនិម្មិតឱ្យបានលម្អិត",
        "សរសេររឿងខ្លីអំពីបច្ចេកវិទ្យា",
        "សង្ខេបអត្ថបទខាងក្រោម៖ បញ្ញាសិប្បនិម្មិតកំពុងផ្លាស់ប្តូរពិភពលោក...",
        "ប្រៀបធៀបការរៀនម៉ាស៊ីននិងការរៀនជ្រៅ",
        "តើអ្វីទៅជាអត្ថប្រយោជន៍នៃការប្រើប្រាស់ AI?",
        "ពន្យល់ពីរបៀបដែល neural network ដំណើរការ",
        "បង្កើតតារាងប្រៀបធៀបភាសាកម្មវិធីផ្សេងៗ",
        "វិភាគផលប៉ះពាល់នៃបច្ចេកវិទ្យាលើសង្គម",
        "សរសេរអត្ថបទអំពីប្រវត្តិនៃកុំព្យូទ័រ",
        "ពន្យល់ពីគោលការណ៍មូលដ្ឋាននៃការសរសេរកូដ",
    ]
    
    # Additional contextual prompts for variety
    KHMER_CONTEXTS = [
        "អំពីកម្ពុជា", "អំពីវប្បធម៌", "អំពីបច្ចេកវិទ្យា", "អំពីការអប់រំ",
        "អំពីសេដ្ឋកិច្ច", "អំពីប្រវត្តិសាស្ត្រ", "អំពីវិទ្យាសាស្ត្រ", "អំពីសង្គម"
    ]
    
    @classmethod
    def generate_prompt_distribution(cls, num_samples):
        """Generate prompts following the specified distribution"""
        prompts = []
        
        # Calculate exact numbers for each category
        num_km_to_en = int(num_samples * config.KHMER_TO_ENGLISH_RATIO)
        num_en_to_km = int(num_samples * config.ENGLISH_TO_KHMER_RATIO)
        num_km_to_km = num_samples - num_km_to_en - num_en_to_km
        
        # Generate Khmer → English (50%)
        for _ in range(num_km_to_en):
            prompt = random.choice(cls.KHMER_TO_ENGLISH)
            prompts.append({
                "prompt": prompt,
                "task_type": "khmer_to_english",
                "category": "translation"
            })
        
        # Generate English → Khmer (30%)
        for _ in range(num_en_to_km):
            prompt = random.choice(cls.ENGLISH_TO_KHMER)
            prompts.append({
                "prompt": prompt,
                "task_type": "english_to_khmer",
                "category": "translation"
            })
        
        # Generate Khmer → Khmer (20%)
        for _ in range(num_km_to_km):
            prompt = random.choice(cls.KHMER_TO_KHMER)
            prompts.append({
                "prompt": prompt,
                "task_type": "khmer_to_khmer",
                "category": "khmer_task"
            })
        
        # Shuffle to mix the distribution
        random.shuffle(prompts)
        
        return prompts

# ============================================================================
# STEP 4: LOAD TEACHER MODEL (QWEN3)
# ============================================================================

def load_teacher_model():
    """Load Qwen3 teacher model for knowledge extraction"""
    print("Loading teacher model (Qwen3)...")
    
    teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.TEACHER_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    FastLanguageModel.for_inference(teacher_model)
    return teacher_model, teacher_tokenizer

# ============================================================================
# STEP 5: GENERATE BILINGUAL TRAINING DATA
# ============================================================================

def generate_bilingual_dataset(teacher_model, teacher_tokenizer, num_samples):
    """Extract bilingual knowledge from teacher model"""
    print(f"Generating {num_samples} bilingual training samples...")
    print(f"Distribution: {config.KHMER_TO_ENGLISH_RATIO*100}% KM→EN, "
          f"{config.ENGLISH_TO_KHMER_RATIO*100}% EN→KM, "
          f"{config.KHMER_TO_KHMER_RATIO*100}% KM→KM")
    
    # Generate prompts with distribution
    prompt_data = BilingualPrompts.generate_prompt_distribution(num_samples)
    
    dataset = []
    
    for item in tqdm(prompt_data, desc="Extracting knowledge"):
        prompt = item["prompt"]
        task_type = item["task_type"]
        
        # Format prompt for Qwen
        messages = [{"role": "user", "content": prompt}]
        text = teacher_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response from teacher
        inputs = teacher_tokenizer([text], return_tensors="pt").to("cuda")
        
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        response = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response.split(prompt)[-1].strip()
        if not response:
            # Fallback if splitting fails
            response = teacher_tokenizer.decode(outputs[0][len(inputs[0]):], 
                                               skip_special_tokens=True).strip()
        
        # Format for training
        training_text = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>"
        
        dataset.append({
            "prompt": prompt,
            "answer": response,
            "text": training_text,
            "task_type": task_type,
            "category": item["category"]
        })
    
    return Dataset.from_list(dataset)

# ============================================================================
# STEP 6: LOAD STUDENT MODEL (PHI3) WITH LORA
# ============================================================================

def load_student_model():
    """Load Phi3 student model with LoRA for efficient training"""
    print("Loading student model (Phi3) with LoRA...")
    
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.STUDENT_MODEL,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    # Add LoRA adapters
    student_model = FastLanguageModel.get_peft_model(
        student_model,
        r=config.LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return student_model, student_tokenizer

# ============================================================================
# STEP 7: TRAINING FUNCTION
# ============================================================================

def train_student_model(student_model, student_tokenizer, dataset):
    """Train Phi3 on bilingual knowledge from Qwen3"""
    print("Starting bilingual distillation training...")
    
    trainer = SFTTrainer(
        model=student_model,
        tokenizer=student_tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=config.OUTPUT_DIR,
            save_strategy="epoch",
            save_total_limit=2,
        ),
    )
    
    # Train
    trainer.train()
    
    return trainer

# ============================================================================
# STEP 8: SAVE MODELS AND DATASET
# ============================================================================

def save_models_and_data(student_model, student_tokenizer, dataset, save_4bit=False):
    """Save trained model and training data"""
    print("Saving models and data...")
    
    # Save dataset with metadata
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    dataset.save_to_disk(f"{config.OUTPUT_DIR}/training_data")
    
    # Save dataset as JSON for inspection
    dataset.to_json(f"{config.OUTPUT_DIR}/training_data.json", force_ascii=False)
    
    # Save LoRA adapters (smallest, most flexible)
    print("Saving LoRA adapters...")
    student_model.save_pretrained(f"{config.OUTPUT_DIR}/lora_adapters")
    student_tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/lora_adapters")
    
    # Save merged model (16-bit) - RECOMMENDED for quality
    print("Saving merged 16-bit model (recommended)...")
    student_model.save_pretrained_merged(
        f"{config.OUTPUT_DIR}/merged_16bit",
        student_tokenizer,
        save_method="merged_16bit",
    )
    
    # Save merged model (4-bit quantized) - OPTIONAL, only if you need it
    if save_4bit:
        print("Saving merged 4-bit model (final deployment only)...")
        student_model.save_pretrained_merged(
            f"{config.OUTPUT_DIR}/merged_4bit",
            student_tokenizer,
            save_method="merged_4bit_forced",  # Use forced to acknowledge accuracy loss
        )
        print("⚠️  4-bit model saved. Note: Some accuracy loss may occur.")
    else:
        print("Skipping 4-bit save. Use save_4bit=True if needed for final deployment.")
    
    print(f"\nModels and data saved to {config.OUTPUT_DIR}")
    print("\nSaved formats:")
    print(f"  ✓ LoRA adapters: {config.OUTPUT_DIR}/lora_adapters (smallest)")
    print(f"  ✓ 16-bit merged: {config.OUTPUT_DIR}/merged_16bit (best quality)")
    if save_4bit:
        print(f"  ✓ 4-bit merged: {config.OUTPUT_DIR}/merged_4bit (deployment)")

# ============================================================================
# STEP 9: COMPREHENSIVE TESTING
# ============================================================================

def test_bilingual_model(model, tokenizer):
    """Test the distilled model on all task types"""
    print("\n" + "="*80)
    print("TESTING BILINGUAL MODEL")
    print("="*80)
    
    FastLanguageModel.for_inference(model)
    
    test_cases = {
        "Khmer → English Translation": [
            "បកប្រែទៅជាភាសាអង់គ្លេស៖ បញ្ញាសិប្បនិម្មិតជាអ្វី?",
            "Translate to English: ការរៀនម៉ាស៊ីនជាផ្នែកមួយនៃ AI",
            "បកប្រែ៖ កម្ពុជាមានប្រវត្តិសាស្ត្រយូរអង្វែង",
        ],
        "English → Khmer Translation": [
            "Translate to Khmer: What is overfitting?",
            "បកប្រែទៅខ្មែរ៖ Machine learning is powerful.",
            "Translate to Khmer: Data is the new oil.",
        ],
        "Khmer Tasks": [
            "សូមពន្យល់ពីបញ្ញាសិប្បនិម្មិត",
            "តើការរៀនម៉ាស៊ីនដំណើរការយ៉ាងណា?",
            "ប្រៀបធៀប AI និង ML",
        ]
    }
    
    for category, prompts in test_cases.items():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")
        
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 80)

# ============================================================================
# STEP 10: DATASET ANALYSIS
# ============================================================================

def analyze_dataset(dataset):
    """Analyze the generated dataset distribution"""
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    task_counts = {}
    for item in dataset:
        task_type = item["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    total = len(dataset)
    print(f"\nTotal samples: {total}")
    print("\nDistribution:")
    for task_type, count in sorted(task_counts.items()):
        percentage = (count / total) * 100
        print(f"  {task_type}: {count} ({percentage:.1f}%)")
    
    # Show sample entries
    print("\n" + "="*80)
    print("SAMPLE ENTRIES")
    print("="*80)
    
    for task_type in task_counts.keys():
        sample = next(item for item in dataset if item["task_type"] == task_type)
        print(f"\nTask Type: {task_type}")
        print(f"Prompt: {sample['prompt']}")
        print(f"Answer: {sample['answer'][:200]}...")
        print("-" * 80)

# ============================================================================
# STEP 11: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Complete bilingual knowledge distillation pipeline"""
    
    print("=" * 80)
    print("BILINGUAL KNOWLEDGE DISTILLATION")
    print("Qwen3-Thinking → Phi3 (Khmer-English Transfer)")
    print("=" * 80)
    
    # Step 1: Load teacher model
    teacher_model, teacher_tokenizer = load_teacher_model()
    
    # Step 2: Generate bilingual training data
    training_dataset = generate_bilingual_dataset(
        teacher_model,
        teacher_tokenizer,
        config.NUM_SAMPLES
    )
    
    # Step 3: Analyze dataset
    analyze_dataset(training_dataset)
    
    # Free teacher model memory
    del teacher_model, teacher_tokenizer
    torch.cuda.empty_cache()
    
    # Step 4: Load student model
    student_model, student_tokenizer = load_student_model()
    
    # Step 5: Train student model
    trainer = train_student_model(student_model, student_tokenizer, training_dataset)
    
    # Step 6: Save models and data
    # By default, only saves LoRA + 16-bit (best quality)
    # Set save_4bit=True only for final deployment
    save_models_and_data(student_model, student_tokenizer, training_dataset, save_4bit=False)
    
    # Step 7: Test the model
    test_bilingual_model(student_model, student_tokenizer)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Models saved to: {config.OUTPUT_DIR}")
    print("=" * 80)

# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def save_final_4bit_model(model_path):
    """
    Convert an existing 16-bit model to 4-bit for final deployment
    This is the recommended workflow to avoid accuracy loss
    
    Usage:
        # After training and saving 16-bit model
        save_final_4bit_model("./phi3_khmer_english_distilled/merged_16bit")
    """
    print("Loading 16-bit model for final 4-bit conversion...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,  # Load in full precision first
    )
    
    print("Converting to 4-bit (final deployment)...")
    model.save_pretrained_merged(
        f"{model_path}_4bit",
        tokenizer,
        save_method="merged_4bit_forced",
    )
    
    print(f"4-bit model saved to: {model_path}_4bit")
    print("⚠️  This 4-bit model is optimized for deployment, not further training.")

def load_trained_model(model_path):
    """Load the distilled model for inference"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """Generate a single response"""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()

# ============================================================================
# EXAMPLE USAGE AFTER TRAINING
# ============================================================================
"""
# ============================================================================
# RECOMMENDED WORKFLOW
# ============================================================================

# 1. Train and save (automatically saves LoRA + 16-bit)
python distillation_training.py

# 2. Test with 16-bit model (best quality)
from distillation_training import load_trained_model, generate_response

model, tokenizer = load_trained_model("./phi3_khmer_english_distilled/merged_16bit")

# Test Khmer → English
response = generate_response(model, tokenizer, 
    "បកប្រែទៅជាភាសាអង់គ្លេស៖ បញ្ញាសិប្បនិម្មិតជាអ្វី?")
print(response)

# Test English → Khmer
response = generate_response(model, tokenizer,
    "Translate to Khmer: What is machine learning?")
print(response)

# Test Khmer → Khmer
response = generate_response(model, tokenizer,
    "សូមពន្យល់ពីបញ្ញាសិប្បនិម្មិត")
print(response)

# 3. ONLY IF NEEDED: Convert to 4-bit for final deployment
from distillation_training import save_final_4bit_model

save_final_4bit_model("./phi3_khmer_english_distilled/merged_16bit")

# 4. Use the 4-bit model (smaller, faster, slight accuracy loss)
model_4bit, tokenizer_4bit = load_trained_model(
    "./phi3_khmer_english_distilled/merged_16bit_4bit"
)

# ============================================================================
# WHAT EACH MODEL IS FOR
# ============================================================================

# lora_adapters/
#   → Smallest size (~100-500MB)
#   → Can be loaded on top of base Phi3 model
#   → Best for sharing or continued training
#   → Load with: FastLanguageModel.from_pretrained + load LoRA adapters

# merged_16bit/
#   → Best quality (~7-8GB)
#   → Full precision, no accuracy loss
#   → Use for testing, evaluation, or further fine-tuning
#   → Recommended for most use cases

# merged_16bit_4bit/ (created separately if needed)
#   → Smallest merged size (~4GB)
#   → Fast inference, lower memory
#   → Some accuracy loss
#   → Use ONLY for final deployment where speed/size matters

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# If you see "RuntimeError: Unsloth: Merging into 4bit will cause..."
# → This is EXPECTED and SAFE
# → The script now only saves 16-bit by default (best quality)
# → Only create 4-bit when you're ready for final deployment
# → Use save_4bit=True in save_models_and_data() or use save_final_4bit_model()

# To force 4-bit save during training (not recommended):
# In main(), change:
# save_models_and_data(student_model, student_tokenizer, training_dataset, save_4bit=True)
"""