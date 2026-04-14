# -*- coding: utf-8 -*-
# Copyright 2025 SGT
# This file is part of SGT, modified from OLMo.
# Licensed under the Apache License, Version 2.0
# Original source: https://github.com/allenai/OLMo
"""
Dynamic Layer Selection Scheduler

Responsible for:
1. Collecting entropy statistics from all layers
2. Selecting layers based on entropy ranking
3. Managing layer selection queue and depth threshold
4. Triggering head selection and loop depth increase
"""
from __future__ import annotations

from typing import List, Dict, Set, Optional, TYPE_CHECKING
import logging

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from .loop_attention import LoopAttention
    from .config import DynamicLoopConfig

__all__ = ["DynamicLayerScheduler"]

log = logging.getLogger(__name__)


class DynamicLayerScheduler:
    """
    Dynamic Layer Selection Scheduler

    Layer selection rules:
    1. Exclude exclude_layers (default: layer_0)
    2. Rank layers by average entropy from high to low, take top_k
    3. Select the deepest layer from top_k (maximum layer_index)
    4. Subsequent layer selection depth cannot exceed the minimum layer_index among selected layers
    5. Cannot select adjacent layers
    6. Maximum of max_selected_layers different layers

    Window mechanism:
    - A window period follows each layer selection/depth increase
    - Head decay occurs during the window period
    - Re-evaluation after window period ends
    """
    
    def __init__(
        self,
        config: DynamicLoopConfig,
        n_layers: int,
        loop_layers: Dict[int, LoopAttention],
    ):
        """
        Args:
            config: DynamicLoopConfig configuration
            n_layers: Total number of model layers
            loop_layers: {layer_idx: LoopAttention} All possible loop layers
        """
        self.config = config
        self.n_layers = n_layers
        self.loop_layers = loop_layers  # All layers have corresponding LoopAttention

        # === Layer selection state ===
        self.selected_layers: List[int] = []  # List of selected layers (in selection order)
        self.depth_threshold: int = n_layers  # Depth threshold (layers deeper than this cannot be selected)
        # self.adjacent_blocked: Set[int] = set()  # Layers blocked by adjacent rule

        # === Window period state ===
        self.current_window_start: int = -1
        self.current_window_end: int = -1
        self.in_window: bool = False  # Whether currently in window period

        # === Phase state ===
        self.phase: str = "vanilla"  # "vanilla", "selecting", "fixed"
        self.last_action_step: int = -1
        
        log.info(
            f"DynamicLayerScheduler initialized: "
            f"first_selection_step={config.first_selection_step}, "
            f"window_size={config.window_size}, "
            f"top_k_layers={config.top_k_layers}, "
            f"max_selected_layers={config.max_selected_layers}, "
            f"max_loop_depth={config.max_loop_depth}, "
            f"exclude_layers={config.exclude_layers}"
        )

    def step(self, global_step: int) -> Dict:
        """
        Called at each training step

        Returns:
            Dictionary containing action information for this step
        """
        action_info = {
            "step": global_step,
            "phase": self.phase,
            "action": None,
            "selected_layer": None,
            "loop_depth_increased": None,
        }
        
        if not self.config.enabled:
            return action_info
        
        # === 1. Update decay for all loop layers ===
        for layer_idx, loop_layer in self.loop_layers.items():
            loop_layer.update_head_decay(global_step)

        # === 2. Check if action is needed ===

        # Not yet time for first layer selection
        if global_step < self.config.first_selection_step:
            return action_info

        # Check if reached first layer selection time
        if global_step == self.config.first_selection_step and self.phase == "vanilla":
            self._do_first_selection(global_step)
            action_info["action"] = "first_selection"
            action_info["selected_layer"] = self.selected_layers[-1] if self.selected_layers else None
            action_info["phase"] = self.phase
            return action_info

        # Structure is fixed, no more changes
        if self.phase == "fixed":
            return action_info

        # In window period, no changes
        if self.in_window and global_step < self.current_window_end:
            return action_info

        # Window period ended, re-evaluate
        if self.in_window and global_step >= self.current_window_end:
            self.in_window = False
            action_info.update(self._evaluate_and_act(global_step))
        
        action_info["phase"] = self.phase
        return action_info

    def _do_first_selection(self, global_step: int):
        """Execute first layer selection"""
        log.info(f"Step {global_step}: Starting first layer selection")
        self.phase = "selecting"

        # Get layer entropy ranking
        layer_entropies = self._get_layer_entropies()
        top_k = self._get_top_k_layers(layer_entropies)

        if not top_k:
            log.warning("No valid layers to select from top_k")
            return

        # Select deepest layer from top_k
        selected_layer = max(top_k)

        # Execute layer selection
        self._select_layer(selected_layer, global_step)

    def _evaluate_and_act(self, global_step: int) -> Dict:
        """
        Evaluation and action after window period ends

        Returns:
            Action information dictionary
        """
        action_info = {
            "action": None,
            "selected_layer": None,
            "loop_depth_increased": None,
        }

        # Get layer entropy ranking
        layer_entropies = self._get_layer_entropies()
        top_k = self._get_top_k_layers(layer_entropies)
        
        log.info(
            f"Step {global_step}: Evaluating after window end. "
            f"Top-k layers: {top_k}, selected_layers: {self.selected_layers}, "
            f"depth_threshold: {self.depth_threshold}"
        )
        
        if not top_k:
            log.info("No valid layers in top_k, structure fixed")
            self._set_structure_fixed()
            return action_info
        
        # Check if selected layers are still in top_k and need depth increase
        for layer_idx in self.selected_layers:
            if layer_idx in top_k:
                loop_layer = self.loop_layers[layer_idx]
                current_depth = loop_layer.current_loop_depth.item()

                if current_depth < self.config.max_loop_depth:
                    # Increase loop depth for this layer
                    success = self._increase_layer_depth(layer_idx, global_step)
                    if success:
                        action_info["action"] = "increase_depth"
                        action_info["loop_depth_increased"] = layer_idx
                        return action_info

        # Check if enough layers selected
        if len(self.selected_layers) >= self.config.max_selected_layers:
            log.info(f"Already selected {len(self.selected_layers)} layers, structure fixed")
            self._set_structure_fixed()
            return action_info

        # Select new layer
        new_layer = self._find_new_layer_to_select(top_k)
        
        if new_layer is not None:
            self._select_layer(new_layer, global_step)
            action_info["action"] = "select_new_layer"
            action_info["selected_layer"] = new_layer
        else:
            log.info("No valid new layer to select, structure fixed")
            self._set_structure_fixed()
        
        return action_info

    def _get_layer_entropies(self) -> Dict[int, float]:
        """Get average entropy for all layers"""
        entropies = {}
        for layer_idx, loop_layer in self.loop_layers.items():
            entropies[layer_idx] = loop_layer.get_layer_mean_entropy()
        return entropies

    def _get_top_k_layers(self, layer_entropies: Dict[int, float]) -> List[int]:
        """
        Get top_k high-entropy layers (excluding exclude_layers)

        Returns:
            List of top_k layers (sorted by entropy from high to low)
        """
        # Filter out excluded layers
        valid_layers = {
            idx: ent for idx, ent in layer_entropies.items()
            if idx not in self.config.exclude_layers
        }

        if not valid_layers:
            return []

        # Sort by entropy
        sorted_layers = sorted(valid_layers.items(), key=lambda x: x[1], reverse=True)

        # Take top_k
        top_k = [idx for idx, _ in sorted_layers[:self.config.top_k_layers]]

        return top_k

    def _find_new_layer_to_select(self, top_k: List[int]) -> Optional[int]:
        """
        Find a new layer to select from top_k

        Rules:
        1. Cannot exceed depth_threshold
        2. Cannot be adjacent layer
        3. Select deepest layer that satisfies conditions
        """
        candidates = []

        for layer_idx in top_k:
            # Already selected
            if layer_idx in self.selected_layers:
                continue

            # Exceeds depth threshold
            if layer_idx > self.depth_threshold:
                continue

            # # Is adjacent layer
            # if layer_idx in self.adjacent_blocked:
            #     continue

            candidates.append(layer_idx)

        if not candidates:
            return None

        # Select the deepest
        return max(candidates)

    def _select_layer(self, layer_idx: int, global_step: int):
        """
        Select a layer for looping

        Args:
            layer_idx: Layer index to select
            global_step: Current step
        """
        log.info(f"Step {global_step}: Selecting layer {layer_idx}")

        # Add to selected list
        self.selected_layers.append(layer_idx)

        # Update depth threshold (minimum among selected layers)
        self.depth_threshold = min(self.selected_layers)

        # Update adjacent blocked list
        # self.adjacent_blocked.add(layer_idx - 1)
        # self.adjacent_blocked.add(layer_idx + 1)

        # Set window period
        window_start = global_step
        window_end = global_step + self.config.window_size
        self.current_window_start = window_start
        self.current_window_end = window_end
        self.in_window = True

        # Let corresponding loop layer select heads
        loop_layer = self.loop_layers[layer_idx]
        loop_layer.select_heads(window_start, window_end)
        
        self.last_action_step = global_step
        
        log.info(
            f"Layer {layer_idx} selected. "
            f"depth_threshold={self.depth_threshold}, "
            # f"adjacent_blocked={self.adjacent_blocked}, "
            f"window=[{window_start}, {window_end}]"
        )

    def _increase_layer_depth(self, layer_idx: int, global_step: int) -> bool:
        """
        Increase loop depth for a layer

        Note: Increasing depth does not require a new decay window since head selection is complete,
        and decay for non-retained heads is already 0 and will not change.

        Returns:
            Whether successful
        """
        loop_layer = self.loop_layers[layer_idx]

        # After increasing depth, need to wait a window period before next decision
        window_start = global_step
        window_end = global_step + self.config.window_size
        
        success = loop_layer.increase_loop_depth(window_start, window_end)
        
        if success:
            self.current_window_start = window_start
            self.current_window_end = window_end
            self.in_window = True
            self.last_action_step = global_step
            
            log.info(
                f"Step {global_step}: Increased loop depth for layer {layer_idx}, "
                f"waiting window=[{window_start}, {window_end}]"
            )
        
        return success

    def finalize_all_entropies(self):
        """Finalize entropy statistics for all layers"""
        for loop_layer in self.loop_layers.values():
            loop_layer.finalize_entropy()

    def sync_entropies(self):
        """Distributed sync of entropy for all layers"""
        if not (dist.is_available() and dist.is_initialized()):
            return
        
        for loop_layer in self.loop_layers.values():
            dist.all_reduce(loop_layer.ema_entropy, op=dist.ReduceOp.AVG)

    def get_scheduler_state(self) -> Dict:
        """Get scheduler state (for logging and checkpoint)"""
        return {
            "phase": self.phase,
            "selected_layers": self.selected_layers.copy(),
            "depth_threshold": self.depth_threshold,
            # "adjacent_blocked": list(self.adjacent_blocked),
            "current_window": [self.current_window_start, self.current_window_end],
            "in_window": self.in_window,
            "last_action_step": self.last_action_step,
            "layer_states": {
                idx: layer.get_loop_state_dict()
                for idx, layer in self.loop_layers.items()
            }
        }

    def load_scheduler_state(self, state_dict: Dict):
        """
        Load scheduler state from checkpoint

        Args:
            state_dict: Scheduler state dictionary
        """
        self.phase = state_dict.get("phase", "vanilla")
        self.selected_layers = state_dict.get("selected_layers", [])
        self.depth_threshold = state_dict.get("depth_threshold", self.n_layers)
        # self.adjacent_blocked = set(state_dict.get("adjacent_blocked", []))

        window = state_dict.get("current_window", [-1, -1])
        self.current_window_start = window[0]
        self.current_window_end = window[1]
        self.in_window = state_dict.get("in_window", False)
        self.last_action_step = state_dict.get("last_action_step", -1)

        # Load state for each layer
        layer_states = state_dict.get("layer_states", {})
        for idx, layer in self.loop_layers.items():
            if str(idx) in layer_states:
                layer.load_loop_state_dict(layer_states[str(idx)])
            elif idx in layer_states:
                layer.load_loop_state_dict(layer_states[idx])
        
        log.info(
            f"Loaded scheduler state: phase={self.phase}, "
            f"selected_layers={self.selected_layers}, "
            f"depth_threshold={self.depth_threshold}"
        )

    def set_all_structure_fixed(self):
        """Set all layer structures as fixed (for inference)"""
        self._set_structure_fixed()

    def _set_structure_fixed(self):
        """
        Internal method: Set structure as fixed

        Called when enough layers are selected during training, and during inference.
        Notifies all LoopAttention layers that structure is fixed.
        """
        self.phase = "fixed"
        for layer in self.loop_layers.values():
            layer.set_structure_fixed(True)
        log.info(f"Structure fixed. Selected layers: {self.selected_layers}")

    def set_inference_mode(self):
        """Set to inference mode"""
        self.set_all_structure_fixed()
        for layer in self.loop_layers.values():
            layer.set_inference_mode()

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics for logging"""
        metrics = {
            "dynamic_loop/phase": {"vanilla": 0, "selecting": 1, "fixed": 2}.get(self.phase, -1),
            "dynamic_loop/num_selected_layers": len(self.selected_layers),
            "dynamic_loop/depth_threshold": self.depth_threshold,
            "dynamic_loop/in_window": float(self.in_window),
        }

        # State for each layer
        for layer_idx, loop_layer in self.loop_layers.items():
            prefix = f"dynamic_loop/layer_{layer_idx}"
            metrics[f"{prefix}/loop_depth"] = loop_layer.current_loop_depth.item()
            metrics[f"{prefix}/mean_entropy"] = loop_layer.get_layer_mean_entropy()

            # Entropy and decay for each head
            for h in range(loop_layer.n_heads):
                metrics[f"{prefix}/head_{h}/entropy"] = loop_layer.ema_entropy[h].item()
                metrics[f"{prefix}/head_{h}/decay"] = loop_layer.head_decay[h].item()

        return metrics