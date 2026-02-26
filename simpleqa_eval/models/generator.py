"""本地模型生成器

使用 Qwen2.5-0.5B（或其他 HuggingFace 模型）进行文本生成，
并通过 logits 计算 self-confidence。
"""

import logging
import torch
import numpy as np
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalModelGenerator:
    """本地 HuggingFace 模型生成器"""

    def __init__(self, model_name: str, device: str = "cpu",
                 max_new_tokens: int = 128):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load(self):
        """加载模型和 tokenizer"""
        if self.model is not None:
            return

        logger.info(f"🔄 加载模型: {self.model_name}")
        logger.info(f"   设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()

        logger.info(f"✅ 模型加载完成")

    def generate_answer(
        self,
        question: str,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> Tuple[str, float]:
        """单次生成答案，返回 (答案文本, self-confidence)。

        Self-confidence = exp(mean(log_probs of generated tokens))
        """
        self.load()

        if seed is not None:
            torch.manual_seed(seed)

        # 构造 prompt
        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # 生成（带 logits 返回）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(temperature, 1e-4),  # 防止 temperature=0 报错
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else 1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 提取生成的 token ids
        generated_ids = outputs.sequences[0, input_len:]
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 计算 self-confidence：对每个生成 token 的 log-prob 取平均再 exp
        confidence = self._compute_confidence(outputs.scores, generated_ids)

        return answer_text, confidence

    def sample_answers(
        self,
        question: str,
        n: int = 10,
        temperature: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """批量采样，优先用 num_return_sequences 一次生成 n 条结果。

        Returns:
            [(答案文本, confidence), ...]
        """
        self.load()
        if n <= 0:
            return []

        do_sample = temperature > 0

        torch.manual_seed(42)

        prompt = self._build_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=max(temperature, 1e-4) if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.95 if do_sample else 1.0,
                num_return_sequences=n if do_sample else 1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # outputs.sequences: (n, seq_len)
        # outputs.scores: tuple of (n, vocab_size) tensors, one per generated position
        results = []
        if not do_sample:
            generated_ids = outputs.sequences[0, input_len:]
            answer_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            confidence = self._compute_confidence(
                outputs.scores, generated_ids, seq_idx=0
            )
            return [(answer_text, confidence) for _ in range(n)]

        for seq_idx in range(n):
            generated_ids = outputs.sequences[seq_idx, input_len:]
            answer_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            confidence = self._compute_confidence(
                outputs.scores, generated_ids, seq_idx
            )
            results.append((answer_text, confidence))

        return results

    def _build_prompt(self, question: str) -> str:
        """构造问答 prompt"""
        # 使用 chat template（如果模型支持），否则用简单 prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question concisely and directly. Give only the answer, no explanation."},
                {"role": "user", "content": question},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        return f"Question: {question}\nAnswer:"

    def _compute_confidence(
        self,
        scores: tuple,
        generated_ids: torch.Tensor,
        seq_idx: int = 0,
    ) -> float:
        """
        从生成 token 的 logits 计算 self-confidence。

        confidence = exp( mean( log_softmax(logits)[token_id] ) )
        即生成序列的几何平均概率。

        Args:
            scores: generate 返回的 scores，每个元素 shape 为 (batch, vocab_size)
            generated_ids: 当前序列的生成 token ids
            seq_idx: 当前序列在 batch 中的索引
        """
        if len(scores) == 0:
            return 0.0

        log_probs = []
        n_tokens = min(len(scores), len(generated_ids))

        for i in range(n_tokens):
            logits = scores[i][seq_idx]  # shape: (vocab_size,)
            log_softmax = torch.log_softmax(logits, dim=-1)
            token_id = generated_ids[i].item()

            # 跳过特殊 token
            if token_id == self.tokenizer.eos_token_id:
                break
            if token_id == self.tokenizer.pad_token_id:
                continue

            log_prob = log_softmax[token_id].item()
            log_probs.append(log_prob)

        if len(log_probs) == 0:
            return 0.0

        mean_log_prob = np.mean(log_probs)
        confidence = float(np.exp(mean_log_prob))

        return confidence
