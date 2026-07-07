from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from transformers.trainer_callback import TrainerCallback
from trl import DPOTrainer


def attach_agc(model, clip: float = 0.01, eps: float = 1e-3):
    def _agc_hook(grad, param):
        if grad is None:
            return grad
        param_norm = param.detach().norm()
        grad_norm = grad.norm()
        max_norm = clip * (param_norm + eps)
        if grad_norm > max_norm:
            grad = grad * (max_norm / (grad_norm + 1e-6))
        return grad

    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g, p=p: _agc_hook(g, p))


class ThresholdStop(TrainerCallback):
    def __init__(self, monitor: str, threshold: float, higher_is_better: bool):
        self.monitor = monitor
        self.threshold = threshold
        self.higher_is_better = higher_is_better

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return
        value = logs[self.monitor]
        stop = (value >= self.threshold) if self.higher_is_better else (value <= self.threshold)
        if stop:
            control.should_training_stop = True
            print(
                f"[ThresholdStop] {self.monitor}={value:.4f} "
                f"crossed {'>=' if self.higher_is_better else '<='} "
                f"{self.threshold} - stopping."
            )


class EarlyStoppingByMetric(TrainerCallback):
    def __init__(
        self,
        monitor: str,
        higher_is_better: bool,
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        self.monitor = monitor
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.monitor not in logs:
            return

        current = logs[self.monitor]
        if self.best is None:
            self.best = current
            return

        improvement = current - self.best if self.higher_is_better else self.best - current
        if improvement > self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                control.should_training_stop = True
                print(f"[EarlyStopping] '{self.monitor}' plateaued (best={self.best:.5f}) - stopping training.")


class FTPOTrainer(DPOTrainer):
    """
    Reference FTPO trainer used by the Antidoom pipeline.

    The only changes here are import-localisation and ASCII log text; the
    collator/loss math is kept identical to the reference implementation.
    """

    def __init__(self, *args, **kwargs):
        train_dataset = kwargs.get("train_dataset")
        processor = kwargs.get("processing_class") or kwargs.get("tokenizer")
        super().__init__(*args, **kwargs)
        train_dataset = train_dataset or getattr(self, "train_dataset", None)
        self.ftpo_max_length = getattr(train_dataset, "max_seq_length", None)
        self.ftpo_pad_token_id = self._resolve_pad_token_id(processor)
        self._ftpo_stored_metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        self.remove_unused_columns = False
        self.data_collator = self.ftpo_collator

    @staticmethod
    def _get_proj(model):
        proj = model.get_output_embeddings()
        if proj is None:
            proj = getattr(model, "lm_head", None)
        if proj is None:
            raise AttributeError("Model lacks both get_output_embeddings() and lm_head.")
        return proj

    def _resolve_pad_token_id(self, processor=None) -> int:
        for obj in (
            processor,
            getattr(self, "processing_class", None),
            getattr(self, "tokenizer", None),
        ):
            pad_token_id = getattr(obj, "pad_token_id", None)
            if pad_token_id is not None:
                return int(pad_token_id)

        padding_value = getattr(self, "padding_value", None)
        if padding_value is not None:
            return int(padding_value)

        config = getattr(getattr(self, "model", None), "config", None)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(config, attr, None)
            if token_id is not None:
                return int(token_id)

        raise AttributeError("Could not determine pad token id for FTPO collator.")

    def _resolve_max_length(self) -> int:
        max_length = getattr(self, "ftpo_max_length", None)
        if max_length is None:
            max_length = getattr(self.args, "max_length", None)
        if max_length is None:
            raise AttributeError("Could not determine max sequence length for FTPO collator.")
        return int(max_length)

    @contextmanager
    def _ftpo_null_ref_context(self, model):
        null_ref_context = getattr(super(), "null_ref_context", None)
        if null_ref_context is not None:
            with null_ref_context():
                yield
            return

        accelerator = getattr(self, "accelerator", None)
        unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
        ref_adapter_name = getattr(self, "ref_adapter_name", None)
        model_adapter_name = getattr(self, "model_adapter_name", None)
        has_disable_adapter = hasattr(unwrapped_model, "disable_adapter")
        should_disable_adapter = (
            (getattr(self, "is_peft_model", False) or has_disable_adapter)
            and not ref_adapter_name
            and has_disable_adapter
        )
        adapter_model = unwrapped_model if hasattr(unwrapped_model, "set_adapter") else model

        with unwrapped_model.disable_adapter() if should_disable_adapter else nullcontext():
            if ref_adapter_name and hasattr(adapter_model, "set_adapter"):
                adapter_model.set_adapter(ref_adapter_name)
            yield
            if ref_adapter_name and hasattr(adapter_model, "set_adapter"):
                adapter_model.set_adapter(model_adapter_name or "default")

    def _store_ftpo_metrics(self, metrics, train_eval: str = "train") -> None:
        store_metrics = getattr(super(), "store_metrics", None)
        if store_metrics is not None:
            store_metrics(metrics, train_eval=train_eval)
            return

        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            else:
                value = float(value)
            self._ftpo_stored_metrics[train_eval][key].append(value)

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        train_eval = "eval" if any(key.startswith("eval") for key in logs) else "train"
        stored = self._ftpo_stored_metrics.get(train_eval, {})
        prefix = "" if train_eval == "train" else f"{train_eval}_"
        for key, values in stored.items():
            if values:
                logs[f"{prefix}{key}"] = sum(values) / len(values)
        stored.clear()
        return super().log(logs, *args, **kwargs)

    def ftpo_collator(self, features):
        pad_id = self.ftpo_pad_token_id
        max_len = self._resolve_max_length()
        batch_sz = len(features)

        prompt_ids = torch.full((batch_sz, max_len), pad_id, dtype=torch.long)
        attention_ms = torch.zeros_like(prompt_ids, dtype=torch.bool)

        for i, feat in enumerate(features):
            seq = torch.tensor(feat["prompt_ids"], dtype=torch.long)
            if seq.size(0) + 1 > max_len:
                raise ValueError(
                    "FTPO sample exceeds max_seq_length; overlength samples must be "
                    "discarded before collation, not truncated"
                )
            prompt_ids[i, -seq.size(0) :] = seq
            attention_ms[i, -seq.size(0) :] = True

        batch = dict(
            prompt_ids=prompt_ids,
            attention_mask=attention_ms,
            rejected_token_id=torch.tensor([f["rejected_token_id"] for f in features]),
        )

        max_c = max(len(f["chosen_ids"]) for f in features)
        chosen_pad = torch.full((batch_sz, max_c), pad_id, dtype=torch.long)
        chosen_mask = torch.zeros_like(chosen_pad, dtype=torch.bool)
        for i, f in enumerate(features):
            ids = torch.tensor(f["chosen_ids"], dtype=torch.long)
            chosen_pad[i, : ids.size(0)] = ids
            chosen_mask[i, : ids.size(0)] = True
        batch.update(chosen_ids=chosen_pad, chosen_mask=chosen_mask)

        return batch

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        lambda_mse_target = getattr(self, "lambda_mse_target", 0.05)
        tau_mse_target = getattr(self, "tau_mse_target", 1.0)
        lambda_mse = getattr(self, "lambda_mse", 0.4)
        clip_epsilon_logits = getattr(self, "clip_epsilon_logits", 2)

        USE_MSE_LOSS = True

        device = next(model.parameters()).device
        ids = inputs["prompt_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        B, L = ids.shape

        seq_len = attn.sum(1)
        pad_off = (L - seq_len).unsqueeze(1)
        arange_L = torch.arange(L, device=ids.device).unsqueeze(0)
        pos_full = (arange_L - pad_off).clamp(min=0)
        pos_full = pos_full.masked_fill(attn == 0, 0)

        outputs = model(
            ids,
            attention_mask=attn,
            position_ids=pos_full,
            use_cache=False,
            return_dict=True,
        )

        logits_last = outputs.logits[:, -1, :]
        logp_all = F.log_softmax(logits_last, dim=-1)

        ch_ids = inputs["chosen_ids"].to(device)
        ch_mask = inputs["chosen_mask"].to(device)
        rejected = inputs["rejected_token_id"].to(device)
        logp_bad = logp_all.gather(-1, rejected.unsqueeze(-1)).squeeze(-1)

        batch_rows = torch.arange(B, device=logp_all.device).unsqueeze(1)
        delta_tok = logits_last[batch_rows, ch_ids] - logits_last.gather(
            -1, rejected.unsqueeze(-1)
        )
        weights = (
            torch.clamp(
                (clip_epsilon_logits - delta_tok) / clip_epsilon_logits,
                0.0,
                1.0,
            )
            * ch_mask
        )

        tau = 1.0
        gap = clip_epsilon_logits - delta_tok
        per_tok_loss = F.softplus(gap / tau)

        chosen_counts = ch_mask.sum(dim=-1).clamp(min=1)
        pref_loss = ((per_tok_loss * weights).sum(dim=-1) / chosen_counts).mean()

        extra_metrics = {}

        if USE_MSE_LOSS:
            with torch.no_grad():
                if self.ref_model is None:
                    with self._ftpo_null_ref_context(model):
                        ref_logits_last = model(
                            ids,
                            attention_mask=attn,
                            position_ids=pos_full,
                            use_cache=False,
                            return_dict=True,
                        ).logits[:, -1, :]
                else:
                    ref_logits_last = self.ref_model(
                        ids,
                        attention_mask=attn,
                        position_ids=pos_full,
                        use_cache=False,
                        return_dict=True,
                    ).logits[:, -1, :]

            tether_mask = torch.ones_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tether_mask[rows[ch_mask], ch_ids[ch_mask]] = False
            tether_mask.scatter_(1, rejected.unsqueeze(-1), False)

            diff = logits_last - ref_logits_last
            mse_elem_raw = (tether_mask * diff.pow(2)).sum() / tether_mask.sum()

            tgt_mask = torch.zeros_like(logits_last, dtype=torch.bool)
            rows = torch.arange(B, device=ch_ids.device).unsqueeze(1).expand_as(ch_ids)
            tgt_mask[rows[ch_mask], ch_ids[ch_mask]] = True
            tgt_mask.scatter_(1, rejected.unsqueeze(-1), True)

            if lambda_mse_target:
                diff_tok = logits_last - ref_logits_last
                diff_tok = diff_tok * tgt_mask
                excess_tok = torch.clamp(diff_tok.abs() - tau_mse_target, min=0.0)
                mse_target_raw = (excess_tok.pow(2)).sum() / tgt_mask.sum()
            else:
                mse_target_raw = logits_last.new_tensor(0.0)

            mse_loss = lambda_mse * mse_elem_raw + lambda_mse_target * mse_target_raw
            loss = pref_loss + mse_loss

            extra_metrics.update(
                {
                    "mse_elem": mse_elem_raw.detach(),
                    "mse_tgt_tokenwise": mse_target_raw.detach(),
                }
            )

        else:
            loss = pref_loss

        lp_chosen = logp_all.gather(-1, ch_ids)
        lp_bad = logp_bad.unsqueeze(-1)

        wins_tok = (lp_chosen > lp_bad) & ch_mask
        frac_win = wins_tok.float().sum(-1) / ch_mask.sum(-1).clamp(min=1e-8)
        chosen_win = frac_win.mean().detach()

        active_delta = delta_tok[ch_mask]
        active_weights = weights[ch_mask]
        margin_win = (
            ((delta_tok >= clip_epsilon_logits) & ch_mask).float().sum()
            / ch_mask.float().sum().clamp(min=1e-8)
        ).detach()
        mean_delta = active_delta.mean().detach()
        median_delta = active_delta.median().detach()
        active_weight = active_weights.mean().detach()

        metrics = {
            "pref_loss": pref_loss.detach(),
            "chosen_win": chosen_win,
            "margin_win": margin_win,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "active_weight": active_weight,
            **extra_metrics,
        }
        self._store_ftpo_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def _prepare_dataset(self, dataset, *args, **_):
        return dataset
