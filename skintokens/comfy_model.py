"""ComfyUI VRAM integration for the SkinTokens model (spec/01, spec/05).

ComfyUI only manages models it knows about. To get the wanted "load on demand,
offload to CPU / evict when the GPU is needed elsewhere" behavior, we wrap the
vendored torch model in a ``comfy.model_patcher.ModelPatcher`` and let
``comfy.model_management`` drive placement and eviction — never hardcoding
``cuda:0``.

``comfy`` is imported lazily inside the functions that need it: this module (and
``nodes.py``) must import cleanly in a GPU-less / ComfyUI-less environment for
local pytest, where these code paths are simply not exercised. The behavioral
contract (spec/01) is: loads on demand, frees when the GPU is needed elsewhere,
survives across runs without leaking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch (via model_loader) at module import time
    from .model_loader import SkinTokensModel


def _ensure_settable_device(model):
    """Make ``model.device`` assignable for ComfyUI's ModelPatcher.

    ComfyUI's ``ModelPatcher.load()`` does ``self.model.device = device`` directly.
    A Lightning ``TokenRig`` exposes ``device`` as a **read-only property**, so that
    assignment raises ``AttributeError: property 'device' ... has no setter``. We
    give just this instance a settable ``device`` by swapping in a runtime subclass
    that overrides the property with a getter+setter (backed by an instance slot,
    falling back to the parameters' device). No vendored files are touched, and
    ``isinstance(model, TokenRig)`` still holds (it's a subclass).
    """
    cls = type(model)
    dev = getattr(cls, "device", None)
    if isinstance(dev, property) and dev.fset is None:
        def _get(self):
            d = self.__dict__.get("_comfy_device")
            if d is not None:
                return d
            try:
                return next(self.parameters()).device
            except StopIteration:
                return None

        def _set(self, value):
            self.__dict__["_comfy_device"] = value

        model.__class__ = type(
            cls.__name__ + "_ComfyPatchable", (cls,), {"device": property(_get, _set)}
        )
    return model


def estimate_model_size(model) -> int:
    """Approximate VRAM footprint of an nn.Module in bytes.

    Sum of parameter + buffer byte sizes. ComfyUI uses this to decide how much
    other-model memory to free before loading; an approximation is fine (a small
    safety margin is added by ComfyUI's own accounting).
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


class SkinTokensModelWrapper:
    """The ``SKINTOKENS_MODEL`` object passed between nodes.

    Holds the loaded :class:`SkinTokensModel` (model + tokenizer + transform) and,
    when running inside ComfyUI, a ``ModelPatcher`` registered with
    ``model_management`` so the model participates in ComfyUI's VRAM lifecycle.
    Outside ComfyUI (local tests) ``patcher`` is ``None`` and the bundle is used
    on whatever device it was loaded on.
    """

    def __init__(self, bundle: SkinTokensModel, patcher=None):
        self.bundle = bundle
        self.patcher = patcher

    @property
    def model(self):
        return self.bundle.model

    def prepare(self) -> SkinTokensModel:
        """Ensure the model is on the compute device; return the ready bundle.

        Inside ComfyUI this asks ``model_management`` to load the patcher onto the
        GPU (evicting/offloading other models as needed) and syncs the bundle's
        device to the compute device. Without ComfyUI it is a no-op and the bundle
        is used as-loaded.
        """
        if self.patcher is None:
            return self.bundle

        import comfy.model_management as mm

        mm.load_models_gpu([self.patcher])
        # Keep the bundle's notion of "device" in sync with where ComfyUI placed
        # the weights, so infer.rig_asset moves inputs to the right device.
        self.bundle.device = self.patcher.load_device
        return self.bundle


def wrap_for_comfy(bundle: SkinTokensModel) -> SkinTokensModelWrapper:
    """Register ``bundle`` with ComfyUI's model management and return a wrapper.

    Builds a ``ModelPatcher`` over the underlying nn.Module with ComfyUI's
    compute/offload devices, so ``model_management`` can offload it to CPU and
    evict it when other workflows need VRAM. The model is expected to have been
    loaded onto the offload (CPU) device; ComfyUI moves it to the GPU on demand
    via :meth:`SkinTokensModelWrapper.prepare`.

    If ``comfy`` is unavailable (local dev), returns a wrapper with no patcher —
    the model is used directly on its loaded device.
    """
    try:
        import comfy.model_management as mm
        from comfy.model_patcher import ModelPatcher
    except Exception:
        return SkinTokensModelWrapper(bundle, patcher=None)

    load_device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()

    _ensure_settable_device(bundle.model)  # ModelPatcher assigns model.device directly
    patcher = ModelPatcher(
        bundle.model,
        load_device=load_device,
        offload_device=offload_device,
        size=estimate_model_size(bundle.model),
    )
    return SkinTokensModelWrapper(bundle, patcher=patcher)
