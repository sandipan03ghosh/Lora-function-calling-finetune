"""Model loading + batched greedy generation. Shared by evaluate / forgetting / serve.

Imports torch at module load, so this file is never imported by the CPU-only unit tests.
"""

from __future__ import annotations

import torch

from .prompts import build_messages


def load_model(
    base: str,
    adapter: str | None = None,
    four_bit: bool = True,
    use_unsloth: bool = True,
    max_seq_length: int = 2048,
):
    """Return (model, tokenizer). If `adapter` is given it is loaded on top of `base`."""
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tok = FastLanguageModel.from_pretrained(
                model_name=adapter or base,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=four_bit,
            )
            FastLanguageModel.for_inference(model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            return model, tok
        except Exception as exc:  # pragma: no cover - depends on runtime
            print(f"[models] Unsloth path unavailable ({exc}); falling back to transformers.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    kwargs: dict = {"torch_dtype": "auto"}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


@torch.inference_mode()
def generate(
    model,
    tok,
    list_of_messages: list[list[dict]],
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> list[str]:
    """Greedy decode. Returns the newly generated text for each input (prompt stripped)."""
    outputs: list[str] = []
    for start in range(0, len(list_of_messages), batch_size):
        chunk = list_of_messages[start : start + batch_size]
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for row in gen:
            outputs.append(tok.decode(row[prompt_len:], skip_special_tokens=True).strip())
    return outputs


def generate_for(model, tok, records: list[dict], **kw) -> list[str]:
    """Convenience: build messages from records and generate."""
    return generate(model, tok, [build_messages(r["query"], r["tools"]) for r in records], **kw)
