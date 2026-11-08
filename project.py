# Waxal ASR Hackathon: Starter Notebook (Gemma 3n + LoRA, Shona 'sna')
# Author: Moath Al-Fawzan
#
# Each "# %%" block is one notebook cell. Run top to bottom.
#
# Legend:
#   [FROM SCREENSHOT]  = copied exactly from your screenshots
#   [RECONSTRUCTED]    = NOT visible in the screenshots; I wrote it so the file
#                        runs end to end. Replace with your notebook's version.
 
# %% [markdown]
# ## Setup  [RECONSTRUCTED]
# %%
from typing import Any
 
import jiwer
import torch
import transformers
import trl
from datasets import Audio, load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
 
LANGUAGE = "sna"
RANDOM_SEED = 42
NUM_VALIDATION_EXAMPLES = 200
MODEL_ID = "google/gemma-3n-E2B-it"
DATASET_NAME = "google/WaxalNLP"   # CHANGE ME
DATASET_CONFIG = "sna_asr"         # CHANGE ME
AUDIO_COL = "audio"                # CHANGE ME
TEXT_COL = "transcription"         # CHANGE ME
PROMPT = "Transcribe this audio."
 
ds = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)
train_ds, val_ds, test_raw = [
    ds[s].cast_column(AUDIO_COL, Audio(sampling_rate=16_000))
    for s in ("train", "validation", "test")
]
 
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Gemma3nForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
)
 
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
 
 
def collator(examples):
    convs = [[
        {"role": "user", "content": [
            {"type": "audio", "audio": ex[AUDIO_COL]["array"]},
            {"type": "text", "text": PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": ex[TEXT_COL]}]},
    ] for ex in examples]
    batch = processor.apply_chat_template(
        convs, add_generation_prompt=False, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch
 
 
# %% [markdown]
# ## 7. Fine-Tune with LoRA (PEFT + TRL SFT)
# %%
# [FROM SCREENSHOT] class docstring; the body is [RECONSTRUCTED]
class _SFTTrainer(trl.SFTTrainer):
    """SFTTrainer with a no-op create_model_card to avoid importlib issues."""
 
    def create_model_card(self, *args, **kwargs):
        return None
 
 
# [FROM SCREENSHOT] first line; the arguments are [RECONSTRUCTED]
training_args = trl.SFTConfig(
    output_dir="outputs/gemma3n-sna-lora",
    max_steps=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    seed=RANDOM_SEED,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to="none",
)
 
# %%
# ------------------------------------------------------------------
# Prepare split datasets
# ------------------------------------------------------------------   [FROM SCREENSHOT]
shuffled_train = (
    train_ds
    .shuffle(buffer_size=1000, seed=RANDOM_SEED)
    .repeat(None)
)
 
# Take a fixed validation slice for reproducibility
val_ds_fixed = val_ds.take(NUM_VALIDATION_EXAMPLES)
 
# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
trainer = _SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=shuffled_train,
    eval_dataset=val_ds_fixed,
    peft_config=peft_config,
)
 
print("✅ Trainer initialised. Starting fine-tuning …")
trainer.train()
print("✅ Fine-tuning complete.")
 
# %% [markdown]
# ## 8. Evaluate: Word Error Rate (WER) & Character Error Rate (CER)
#
# WER and CER are the standard metrics for ASR. Lower is better.
#
# - **WER** = (substitutions + deletions + insertions) / total reference words
# - **CER** = same but at the character level (important for morphologically rich languages)
 
# %%
# [FROM SCREENSHOT] `import jiwer` and the signature of transcribe_batch;
# the body and run_evaluation are [RECONSTRUCTED]
import jiwer
 
 
def transcribe_batch(
    batch: dict[str, list[Any]],
    model: transformers.Gemma3nForConditionalGeneration,
    proc: transformers.AutoProcessor,
    max_new_tokens: int = 128,
) -> list[str]:
    convs = [[{"role": "user", "content": [
        {"type": "audio", "audio": a["array"]},
        {"type": "text", "text": PROMPT}]}] for a in batch[AUDIO_COL]]
    inputs = proc.apply_chat_template(
        convs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in proc.batch_decode(new, skip_special_tokens=True)]
 
 
def run_evaluation(test_ds, model, proc, batch_size=8):
    model.eval()
    proc.tokenizer.padding_side = "left"
    refs, hyps = [], []
    for batch in test_ds.iter(batch_size=batch_size):
        hyps.extend(transcribe_batch(batch, model, proc))
        refs.extend(batch[TEXT_COL])
    refs = [r.lower().strip() for r in refs]
    hyps = [h.lower().strip() for h in hyps]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}
 
 
def format_batch(batch):  # [RECONSTRUCTED] pass-through formatter
    return batch
 
 
# %%
# [FROM SCREENSHOT] the `.map(...)` line and everything after it
test_ds = (
    test_raw
    .map(format_batch, batched=True, batch_size=32)
)
 
print("Running evaluation on test split …")
metrics = run_evaluation(test_ds, model, processor)
print(f"\n📊 Results on {LANGUAGE} test set:")
print(f"   WER : {metrics['wer']:.2%}")
print(f"   CER : {metrics['cer']:.2%}")
 
