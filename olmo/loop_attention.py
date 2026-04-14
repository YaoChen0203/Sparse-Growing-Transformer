# -*- coding: utf-8 -*-
# Copyright 2025 SGT
# This file is part of SGT, modified from OLMo.
# Licensed under the Apache License, Version 2.0
# Original source: https://github.com/allenai/OLMo
"""
OLMo Dynamic Loop Attention Layer Implementation (Revised)

Core revisions:
1. max_loop_depth represents additional loop count (excluding original forward)
2. Entropy for first forward is computed in OLMoSequentialBlock and passed in
3. LoopAttention only handles additional loops
4. Supports stopping entropy computation after structure is fixed
"""
from typing import List, Dict, Optional
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoopAttention",
]

log = logging.getLogger(__name__)


class LoopAttention(nn.Module):
    """
    Loop Attention Module

    Important:
    - This module only handles **additional loops**
    - The first forward is completed in OLMoSequentialBlock
    - current_loop_depth=0 means no additional loop
    - current_loop_depth=1 means 1 additional loop (total 2 forwards)
    - current_loop_depth=2 means 2 additional loops (total 3 forwards)
    """
    
    def __init__(
        self,
        cfg,
        cache,
        ref_block: nn.Module,
        layer_idx: int,
        max_loop_depth: int = 3,
        num_keep_heads: int = 2,
        decay_type: str = "cosine",
        ema_decay: float = 0.5,
    ):
        """
        Args:
            cfg: Model configuration
            cache: BufferCache for caching attention bias
            ref_block: Reference OLMoSequentialBlock for parameter sharing
            layer_idx: Layer index
            max_loop_depth: Maximum additional loop count (excluding original forward)
            num_keep_heads: Number of heads to retain
            decay_type: Decay type ("linear", "cosine", "exponential")
            ema_decay: EMA decay coefficient
        """
        super().__init__()
        self.cfg = cfg
        self.__cache = cache
        self.layer_idx = layer_idx
        self.max_loop_depth = max_loop_depth
        self.num_keep_heads = num_keep_heads
        self.decay_type = decay_type
        self.ema_decay = ema_decay
        
        # === Basic dimensions ===
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # === Parameter sharing: use ref_block's parameters ===
        # Note: Cannot use self.xxx = ref_block.xxx directly, as it would register to _modules
        # causing optimizer to count parameters twice and state_dict to save duplicates
        # Use __dict__ direct assignment to bypass nn.Module's __setattr__
        self.__dict__["qkv_proj"] = ref_block.att_proj
        self.__dict__["attn_out"] = ref_block.attn_out
        self.__dict__["attn_norm"] = ref_block.attn_norm

        # RoPE also uses __dict__
        if cfg.rope:
            self.__dict__["rotary_emb"] = ref_block.rotary_emb

        # === Dynamic state ===
        # Current additional loop depth (0 = no additional loop)
        self.register_buffer("current_loop_depth", torch.tensor(0, dtype=torch.long))

        # Head retention mask
        self.register_buffer("head_keep_mask", torch.zeros(self.n_heads, dtype=torch.bool))

        # Head decay coefficients
        self.register_buffer("head_decay", torch.ones(self.n_heads))

        # Selected head indices
        self.register_buffer("kept_head_indices", torch.zeros(num_keep_heads, dtype=torch.long))

        # === Entropy statistics ===
        # Entropy storage: index 0 = first forward (passed from external), index 1+ = additional loops
        # shape: (max_loop_depth + 1, n_heads)
        self.register_buffer(
            "entropy_per_loop",
            torch.zeros(max_loop_depth + 1, self.n_heads)
        )

        # EMA accumulated entropy
        self.register_buffer("ema_entropy", torch.zeros(self.n_heads))

        # Entropy accumulator
        self.register_buffer("entropy_accumulator", torch.zeros(max_loop_depth + 1, self.n_heads))
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long))

        # === Decay window state ===
        self.register_buffer("decay_start_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("decay_end_step", torch.tensor(-1, dtype=torch.long))

        # === Flags ===
        self.head_selection_done = False
        self.structure_fixed = False  # Whether structure is fixed
        self.compute_entropy_after_fixed = False  # Whether to continue computing entropy after fixed

        # Cache
        self._head_decay_view = None
        
        # fused_dims for QKV split
        head_dim = cfg.d_model // cfg.n_heads
        self.fused_dims = (
            cfg.d_model,
            cfg.effective_n_kv_heads * head_dim,
            cfg.effective_n_kv_heads * head_dim,
        )

    def set_compute_entropy_after_fixed(self, value: bool):
        """Set whether to continue computing entropy after structure is fixed"""
        self.compute_entropy_after_fixed = value

    def set_structure_fixed(self, fixed: bool):
        """Set whether structure is fixed"""
        self.structure_fixed = fixed

    # =========================================================================
    # Shape transformation utilities
    # =========================================================================

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, Hd)"""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Hd) -> (B, T, D)"""
        B, H, T, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Hd)

    # =========================================================================
    # Entropy computation
    # =========================================================================
    
    @staticmethod
    def compute_entropy_from_weights(attn_weights_last: torch.Tensor) -> torch.Tensor:
        """
        Compute per-head entropy from attention weights (static method, can be called externally)

        Args:
            attn_weights_last: (B, H, Tk) attention weights for the last query position

        Returns:
            (H,) normalized entropy for each head
        """
        B, H, Tk = attn_weights_last.shape
        probs = attn_weights_last.clamp_min(1e-12)
        log_probs = probs.log()
        ent = -(probs * log_probs).sum(dim=-1).mean(dim=0)  # (H,)

        # Normalize
        norm = math.log(max(Tk, 1))
        return ent / norm if norm > 0 else ent

    @torch.no_grad()
    def accumulate_first_forward_entropy(self, attn_weights_last: torch.Tensor):
        """
        Accumulate entropy from first forward (called from OLMoSequentialBlock)

        Args:
            attn_weights_last: (B, H, Tk) attention weights for the last query position
        """
        # Check if entropy computation is needed
        if self.structure_fixed and not self.compute_entropy_after_fixed:
            return

        entropy = self.compute_entropy_from_weights(attn_weights_last)
        self.entropy_accumulator[0] += entropy
        self.entropy_count += 1

    @torch.no_grad()
    def _accumulate_loop_entropy(self, attn_weights: torch.Tensor, loop_idx: int):
        """
        Accumulate entropy from additional loops (internal call)

        Args:
            attn_weights: (B, H, Tq, Tk) attention weights
            loop_idx: Additional loop index (1-based), 1 means first additional loop
        """
        if self.structure_fixed and not self.compute_entropy_after_fixed:
            return

        # Only take the last query position to save memory
        attn_weights_last = attn_weights[:, :, -1, :]  # (B, H, Tk)
        entropy = self.compute_entropy_from_weights(attn_weights_last)
        self.entropy_accumulator[loop_idx] += entropy

    @torch.no_grad()
    def finalize_entropy(self):
        """
        Finalize entropy accumulation, compute average and update EMA
        """
        if self.entropy_count == 0:
            return

        if self.structure_fixed and not self.compute_entropy_after_fixed:
            # Structure is fixed and no need to compute entropy, reset accumulator and return
            self.entropy_accumulator.zero_()
            self.entropy_count.zero_()
            return

        count = self.entropy_count.float()

        # Compute average entropy
        current_depth = self.current_loop_depth.item()
        for loop_idx in range(current_depth + 1):
            self.entropy_per_loop[loop_idx] = self.entropy_accumulator[loop_idx] / count

        # Update EMA entropy
        # Rule: Retained heads use entropy from last loop, decayed heads use entropy from first forward
        if self.head_selection_done:
            for h in range(self.n_heads):
                if self.head_keep_mask[h]:
                    # Retained head: use entropy from last loop (index = current_depth)
                    instant_entropy = self.entropy_per_loop[current_depth, h]
                else:
                    # Decaying head: use entropy from first forward (index = 0)
                    instant_entropy = self.entropy_per_loop[0, h]

                if self.ema_decay == 0.0:
                    self.ema_entropy[h] = instant_entropy
                else:
                    self.ema_entropy[h] = (
                        self.ema_decay * self.ema_entropy[h] +
                        (1.0 - self.ema_decay) * instant_entropy
                    )
        else:
            # Heads not yet selected, use entropy from first forward
            instant_entropy = self.entropy_per_loop[0]
            if self.ema_decay == 0.0:
                self.ema_entropy.copy_(instant_entropy)
            else:
                self.ema_entropy.mul_(self.ema_decay).add_(instant_entropy * (1.0 - self.ema_decay))

        # Reset accumulator
        self.entropy_accumulator.zero_()
        self.entropy_count.zero_()

    def get_layer_mean_entropy(self) -> float:
        """Get layer mean entropy (for layer selection)"""
        return self.ema_entropy.mean().item()

    # =========================================================================
    # Head selection and decay
    # =========================================================================

    @torch.no_grad()
    def select_heads(self, window_start: int, window_end: int):
        """Select heads to retain"""
        if self.head_selection_done:
            log.info(f"Layer {self.layer_idx}: Head selection already done, skipping")
            return

        # Select top num_keep_heads heads with highest entropy
        _, indices = torch.sort(self.ema_entropy, descending=True)
        keep_indices = indices[:self.num_keep_heads]

        self.head_keep_mask.zero_()
        for idx in keep_indices:
            self.head_keep_mask[idx] = True

        self.kept_head_indices.copy_(keep_indices)

        self.decay_start_step.fill_(window_start)
        self.decay_end_step.fill_(window_end)

        # Set additional loop depth to 1
        self.current_loop_depth.fill_(1)

        self.head_selection_done = True
        
        log.info(
            f"Layer {self.layer_idx}: Selected heads {keep_indices.tolist()} "
            f"(entropy: {[f'{e:.4f}' for e in self.ema_entropy[keep_indices].tolist()]}), "
            f"decay window: [{window_start}, {window_end}]"
        )

    @torch.no_grad()
    def increase_loop_depth(self, window_start: int = -1, window_end: int = -1) -> bool:
        """
        Increase additional loop depth

        Note: Does not reset decay coefficients when increasing depth!
        Non-retained heads have already decayed to 0 after first head selection,
        subsequent depth increase only lets retained heads loop more times

        Args:
            window_start: Optional, for scheduler recording (does not affect decay)
            window_end: Optional, for scheduler recording (does not affect decay)
        """
        current = self.current_loop_depth.item()
        if current >= self.max_loop_depth:
            log.info(f"Layer {self.layer_idx}: Already at max loop depth {self.max_loop_depth}")
            return False
        
        self.current_loop_depth.fill_(current + 1)
        
        log.info(
            f"Layer {self.layer_idx}: Increased loop depth to {current + 1}"
        )
        return True

    @torch.no_grad()
    def update_head_decay(self, global_step: int):
        """Update head decay coefficients"""
        if not self.head_selection_done:
            return
        
        start = self.decay_start_step.item()
        end = self.decay_end_step.item()
        
        if start < 0 or global_step < start:
            return
        
        if global_step >= end:
            for h in range(self.n_heads):
                if not self.head_keep_mask[h]:
                    self.head_decay[h] = 0.0
        else:
            progress = (global_step - start) / (end - start)
            progress = min(max(progress, 0.0), 1.0)
            
            if self.decay_type == "linear":
                decay = 1.0 - progress
            elif self.decay_type == "cosine":
                decay = 0.5 * (1 + math.cos(math.pi * progress))
            elif self.decay_type == "exponential":
                k = -math.log(1e-8)
                decay = math.exp(-k * progress)
            else:
                decay = 1.0 - progress
            
            decay = max(decay, 0.0)
            
            for h in range(self.n_heads):
                if self.head_keep_mask[h]:
                    self.head_decay[h] = 1.0
                else:
                    self.head_decay[h] = decay
        
        self._head_decay_view = None

    def is_decay_window_complete(self, global_step: int) -> bool:
        """Check if current decay window is complete"""
        end = self.decay_end_step.item()
        return end >= 0 and global_step >= end

    # =========================================================================
    # Forward - Only handles additional loops
    # =========================================================================
    
    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass - Only executes additional loops

        Note:
        - If current_loop_depth == 0, returns input directly without any computation
        - Entropy for first forward should be computed externally (in OLMoSequentialBlock)
          and passed via accumulate_first_forward_entropy()

        Args:
            x: Input tensor (B, T, D) - Already went through first forward
            is_causal: Whether to use causal mask

        Returns:
            output: Output tensor (B, T, D)
        """
        loop_depth = self.current_loop_depth.item()

        # No additional loop needed
        if loop_depth == 0:
            return x

        # Execute additional loops
        h = x

        for loop_iter in range(loop_depth):
            # loop_iter: 0, 1, 2, ... (additional loop index)
            # Corresponding entropy index: 1, 2, 3, ... (0 is first forward)
            entropy_idx = loop_iter + 1
            
            # === Norm ===
            if not self.cfg.norm_after:
                h_norm = self.attn_norm(h)
            else:
                h_norm = h
            
            # === QKV Projection ===
            qkv = self.qkv_proj(h_norm)
            q, k, v = qkv.split(self.fused_dims, dim=-1)
            q, k, v = map(self._shape, (q, k, v))
            
            # === RoPE ===
            if self.cfg.rope:
                q, k = self.rotary_emb(q, k)
            
            # === Attention ===
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            
            if is_causal:
                from .model import get_causal_attention_bias
                query_len, key_len = q.shape[-2], k.shape[-2]
                attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)
                attn_bias = attn_bias[:, :, :query_len, :key_len]
                attn_weights = attn_weights + attn_bias
            
            attn_weights = F.softmax(attn_weights, dim=-1)

            # === Collect entropy (during training) ===
            if self.training:
                self._accumulate_loop_entropy(attn_weights.detach(), entropy_idx)

            # === Weighted sum ===
            out = torch.matmul(attn_weights, v)

            # === Apply head decay ===
            if self._head_decay_view is None:
                self._head_decay_view = self.head_decay.view(1, self.n_heads, 1, 1)
            out = out * self._head_decay_view
            
            # === Output projection ===
            attn_out = self.attn_out(self._merge(out))
            
            if self.cfg.norm_after:
                attn_out = self.attn_norm(attn_out)
            
            # === Residual ===
            h = h + attn_out
        
        return h

    # =========================================================================
    # State query, save and load
    # =========================================================================

    def get_state_info(self) -> Dict:
        """Get current state info (for logging)"""
        return {
            "layer_idx": self.layer_idx,
            "current_loop_depth": self.current_loop_depth.item(),
            "max_loop_depth": self.max_loop_depth,
            "head_selection_done": self.head_selection_done,
            "structure_fixed": self.structure_fixed,
            "kept_heads": self.kept_head_indices.tolist() if self.head_selection_done else [],
            "head_decay": self.head_decay.tolist(),
            "ema_entropy": self.ema_entropy.tolist(),
            "decay_window": [self.decay_start_step.item(), self.decay_end_step.item()],
        }

    def get_loop_state_dict(self) -> Dict:
        """
        Get additional state to save (non-buffer Python attributes)

        Returns:
            State dictionary for saving to checkpoint
        """
        return {
            "head_selection_done": self.head_selection_done,
            "structure_fixed": self.structure_fixed,
            "compute_entropy_after_fixed": self.compute_entropy_after_fixed,
        }

    def load_loop_state_dict(self, state_dict: Dict):
        """
        Load additional state

        Args:
            state_dict: State dictionary loaded from checkpoint
        """
        self.head_selection_done = state_dict.get("head_selection_done", False)
        self.structure_fixed = state_dict.get("structure_fixed", False)
        self.compute_entropy_after_fixed = state_dict.get("compute_entropy_after_fixed", False)

        # Infer state from buffer data
        if self.current_loop_depth.item() > 0 and not self.head_selection_done:
            # Buffer has loop depth but flag not set, indicates loading from old checkpoint
            self.head_selection_done = True
            log.info(f"Layer {self.layer_idx}: Inferred head_selection_done=True from buffer")

    def set_inference_mode(self):
        """
        Set to inference mode

        Called during inference to ensure:
        1. No entropy computation
        2. Fixed decay coefficients
        """
        self.structure_fixed = True
        self.compute_entropy_after_fixed = False

        if self.current_loop_depth.item() > 0 and not self.head_selection_done:
            # Buffer has loop depth but flag not set, indicates loading from old checkpoint
            self.head_selection_done = True

        # Ensure non-retained heads have decay of 0
        if self.head_selection_done:
            for h in range(self.n_heads):
                if not self.head_keep_mask[h]:
                    self.head_decay[h] = 0.0
            self._head_decay_view = None