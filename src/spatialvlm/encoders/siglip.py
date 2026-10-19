"""SigLIP2-SO400M/16-NaFlex encoder with multi-layer feature extraction.

Architecture notes:
  - Model: google/siglip2-so400m-patch16-naflex  (⚠ VERIFY model ID at runtime)
  - Input resolution: 384×384 → patch_size=16 → 24×24 = 576 patches (exact, no edge artefacts)
  - SigLIP does NOT have a CLS token — encoder outputs are pure patch tokens
  - Extract intermediate features at layers {9, 18, 27} (evenly-spaced thirds)
  - Channel-concatenate 3 × hidden_size → [B, 576, 3456]
  - All encoder parameters are frozen

Multi-layer extraction rationale (hypothesis H1c):
  Lower layers capture low-level structure (edges, textures);
  middle layers capture mid-level features (surfaces, objects);
  final layer captures semantic context. Concatenating all three
  gives richer spatial features than final-layer-only extraction.

Implementation note on hooks:
  We use register_forward_hook on each transformer encoder block.
  The hook captures the block's output hidden states, which are the
  residual stream activations AFTER the block's attention + MLP.
  These are the semantically richest per-patch representations at each depth.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def _find_encoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the transformer encoder layer list in a SigLIP vision model.

    Tries common HuggingFace structural patterns for SigLIP variants.

    Returns
    -------
    layers : nn.ModuleList
        The list of transformer encoder blocks.

    Raises
    ------
    RuntimeError
        If no encoder layer list is found.
    """
    # Pattern 1: model.vision_model.encoder.layers  (SigLIP, SigLIP2)
    # Pattern 2: model.encoder.layers               (some variants)
    candidate_paths = [
        ["vision_model", "encoder", "layers"],
        ["vision_model", "encoder", "layer"],
        ["encoder", "layers"],
        ["encoder", "layer"],
    ]
    for path in candidate_paths:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            continue

    raise RuntimeError(
        f"Cannot find encoder layers in {type(model).__name__}. "
        "Expected attribute path like vision_model.encoder.layers. "
        "Check the model architecture with: print(model)"
    )


def _call_loader(
    loader: Callable[..., Any],
    model_id: str,
    **kwargs: Any,
) -> Any:
    """Call loader with supported kwargs only."""
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(model_id, **kwargs)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return loader(model_id, **kwargs)

    supported_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return loader(model_id, **supported_kwargs)


class SigLIP2Encoder(nn.Module):
    """SigLIP2-SO400M encoder with multi-layer feature extraction.

    Extracts features from intermediate transformer layers and concatenates
    them along the channel dimension.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. ⚠ VERIFY before use.
    extract_layers : list of int
        1-indexed layer numbers to extract from (e.g. [9, 18, 27]).
        Must be within [1, num_hidden_layers].
    device : torch.device
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch16-naflex",
        extract_layers: list[int] | None = None,
        device: torch.device | None = None,
        lazy_load: bool = False,
        local_files_only: bool = False,
        config_loader: Callable[..., Any] = AutoConfig.from_pretrained,
        model_loader: Callable[..., nn.Module] = AutoModel.from_pretrained,
    ) -> None:
        super().__init__()

        if extract_layers is None:
            extract_layers = [9, 18, 27]

        if device is None:
            device = torch.device("cpu")

        self._model_id = model_id
        self._device = device
        self._local_files_only = local_files_only
        self._model_loader = model_loader

        # Introspect architecture from config
        # SigLIP2 wraps SigLIP vision config; try common attribute paths
        cfg = _call_loader(config_loader, model_id, local_files_only=local_files_only)
        vision_cfg = getattr(cfg, "vision_config", cfg)
        self._hidden_size: int = int(vision_cfg.hidden_size)  # ⚠ verified from config
        self._num_layers: int = int(vision_cfg.num_hidden_layers)  # ⚠ verified from config
        self._patch_size: int = int(vision_cfg.patch_size)  # ⚠ verified from config
        self._image_size: int = int(getattr(vision_cfg, "image_size", 384))  # ⚠ verified

        # Validate extract_layers against actual model depth
        for layer_idx in extract_layers:
            if not (1 <= layer_idx <= self._num_layers):
                raise ValueError(
                    f"extract_layer {layer_idx} is out of range "
                    f"[1, {self._num_layers}] for {model_id}"
                )

        self._extract_layers = sorted(extract_layers)
        self._model: nn.Module | None = None

        # Expected patch count (derived from verified config values)
        patches_per_side = self._image_size // self._patch_size
        self._n_patches: int = patches_per_side * patches_per_side

        # NaFlex detection — updated at load_model() time by inspecting patch_embedding
        self._is_naflex: bool = False

        # Register hooks to capture intermediate features
        # Keys: 0-indexed layer number
        self._hook_outputs: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        if not lazy_load:
            self.load_model()

    def load_model(self) -> None:
        """Load SigLIP weights and register extraction hooks."""
        if self._model is not None:
            return

        model = _call_loader(
            self._model_loader,
            self._model_id,
            local_files_only=self._local_files_only,
        )
        model = model.to(self._device)

        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        self._model = model
        encoder_layers = _find_encoder_layers(self._model)
        for layer_1idx in self._extract_layers:
            layer_0idx = layer_1idx - 1
            hook = encoder_layers[layer_0idx].register_forward_hook(self._make_hook(layer_0idx))
            self._hooks.append(hook)

        # Detect NaFlex: patch_embedding is nn.Linear (expects pre-patchified input)
        # vs standard SigLIP: patch_embedding is nn.Conv2d (accepts raw [B, C, H, W])
        try:
            patch_emb = self._model.vision_model.embeddings.patch_embedding
            self._is_naflex = isinstance(patch_emb, nn.Linear)
        except AttributeError:
            self._is_naflex = False

    def _make_hook(self, layer_0idx: int):
        """Create a forward hook that stores hidden states for the given layer."""

        def hook(module: nn.Module, input: tuple, output) -> None:  # noqa: ARG001
            # Transformer blocks typically return (hidden_states,) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._hook_outputs[layer_0idx] = hidden

        return hook

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract multi-layer features from SigLIP2.

        Parameters
        ----------
        pixel_values : Tensor[B, N, patch_dim] or Tensor[B, 3, H, W]
            NaFlex models require checkpoint-native, pre-patchified values from
            ``Siglip2ImageProcessor``. Conventional SigLIP variants accept normalized BCHW.

        Returns
        -------
        features : Tensor[B, n_patches, n_layers × hidden_size]
            Channel-concatenated features from all extraction layers.
            For default config: [B, 576, 3456]  (576 = 24², 3456 = 3 × 1152)
        """
        self.load_model()
        if self._model is None:
            raise RuntimeError("SigLIP model is unavailable. Call load_model() before forward().")

        self._hook_outputs.clear()
        # Call vision_model directly — the full Siglip2Model also runs a text encoder
        # which requires input_ids. We only need vision features + our hooks.
        vision_model = getattr(self._model, "vision_model", self._model)
        with torch.no_grad():
            if self._is_naflex:
                if pixel_values.ndim != 3:
                    raise ValueError(
                        "SigLIP2 NaFlex requires official pre-patchified [B,N,patch_dim] values; "
                        "raw BCHW input is intentionally rejected."
                    )
                if attention_mask is None or spatial_shapes is None:
                    raise ValueError(
                        "SigLIP2 NaFlex requires pixel_attention_mask and spatial_shapes from "
                        "the checkpoint-native processor."
                    )
                if attention_mask.shape != pixel_values.shape[:2]:
                    raise ValueError("SigLIP2 pixel attention mask does not match patch values.")
                if spatial_shapes.shape != (pixel_values.shape[0], 2):
                    raise ValueError("SigLIP2 spatial_shapes must be [B,2].")
                valid_counts = attention_mask.to(dtype=torch.int64).sum(dim=1)
                shape_products = spatial_shapes[:, 0] * spatial_shapes[:, 1]
                if not torch.equal(valid_counts.cpu(), shape_products.to(dtype=torch.int64).cpu()):
                    raise ValueError("SigLIP2 valid patch counts do not match spatial_shapes.")
                _ = vision_model(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    spatial_shapes=spatial_shapes,
                )
            else:
                if pixel_values.ndim != 4:
                    raise ValueError("Conventional SigLIP input must be [B,3,H,W].")
                _ = vision_model(pixel_values=pixel_values)

        # Collect features in layer order and strip CLS token if present
        features = []
        for layer_1idx in self._extract_layers:
            layer_0idx = layer_1idx - 1
            h = self._hook_outputs[layer_0idx]  # [B, T, hidden_size]

            # The current SVA contract is fixed at exactly n_patches. Padding is
            # rejected by preprocessing rather than truncated silently here.
            n_tokens = h.shape[1]
            if n_tokens == self._n_patches + 1:
                # CLS token at position 0 — strip it
                h = h[:, 1:, :]  # [B, n_patches, hidden_size]
            elif n_tokens != self._n_patches:
                raise RuntimeError(
                    f"Token count {n_tokens} at layer {layer_1idx} does not match expected "
                    f"{self._n_patches}. Check the pinned processor and SVA token contract."
                )

            features.append(h)  # [B, n_patches, hidden_size]

        # Concatenate along channel dimension
        return torch.cat(features, dim=-1)  # [B, n_patches, n_layers × hidden_size]

    def train(self, mode: bool = True) -> SigLIP2Encoder:
        """Keep the frozen pretrained encoder deterministic during model training."""
        super().train(mode)
        if self._model is not None:
            self._model.eval()
        return self

    def __del__(self) -> None:
        # Clean up hooks when the encoder is garbage collected
        for hook in getattr(self, "_hooks", []):
            hook.remove()

    @property
    def out_dim(self) -> int:
        """Output channel dimension = n_extract_layers × hidden_size."""
        return len(self._extract_layers) * self._hidden_size

    @property
    def n_patches(self) -> int:
        """Number of spatial patch tokens."""
        return self._n_patches
