"""Small-model local inference; no CUDA dependency.

On Apple Silicon, select MPS with float16 weights. Use float32 on CPU.
This is an inference path, not an implementation of the upstream editors.
"""

import time


def select_device(torch, requested="auto"):
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_ok = bool(mps_backend and mps_backend.is_available())
    cuda_ok = bool(torch.cuda.is_available())
    if requested == "auto":
        return "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")
    if requested == "mps" and not mps_ok:
        raise RuntimeError("MPS unavailable. Try --device cpu or check your PyTorch build.")
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA unavailable. Try --device auto or cpu.")
    return requested


class LocalModel:
    def __init__(self, model_id, revision="main", device="auto"):
        # Lazy imports: task validation and unit tests work without PyTorch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = select_device(torch, device)
        self.model_id = model_id
        dtype = torch.float16 if self.device in {"mps", "cuda"} else torch.float32

        print(f"Loading {model_id}@{revision} on {self.device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    def generate(self, system_prompt, user_prompt, max_new_tokens=48):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        if self.device == "mps":
            self.torch.mps.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device == "mps":
            self.torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, len(new_tokens), elapsed
