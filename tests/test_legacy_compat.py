from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fact_tokenizer.data import FACTPairedNPYDataset
from fact_tokenizer.model import FACTTokenizer


def small_legacy_model() -> FACTTokenizer:
    return FACTTokenizer(
        model_dim=8,
        dino_dim=8,
        latent_dim=4,
        private_dim=4,
        num_latents=4,
        num_action_slots=2,
        num_private_slots=1,
        patch_size=2,
        enc_blocks=1,
        dec_blocks=1,
        num_heads=2,
        backbone="mock",
        max_time=2,
        max_tokens=32,
    )


def test_legacy_mmap_forward_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    count = 3
    rng = np.random.default_rng(8)
    for view in ("ego", "exo"):
        np.save(tmp_path / f"{view}.npy", rng.integers(0, 255, (count, 2, 8, 8, 3), dtype=np.uint8))
    np.save(tmp_path / "sample_id.npy", np.asarray([f"s{index}" for index in range(count)]))
    np.save(tmp_path / "take_uid.npy", np.asarray(["a", "a", "b"]))
    np.save(tmp_path / "timestamp.npy", np.arange(count, dtype=np.float32))
    dataset = FACTPairedNPYDataset(tmp_path, resize=8)
    assert all(isinstance(array, np.memmap) for array in dataset._videos.values())
    batch = {
        view: {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
            for key, value in dataset[0][view].items()
        }
        for view in ("ego", "exo")
    }
    torch.manual_seed(3)
    model = small_legacy_model().eval()
    with torch.inference_mode():
        first = model(batch)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = small_legacy_model().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    with torch.inference_mode():
        second = restored(batch)
    assert set(first["views"]) == {"ego", "exo"}
    assert set(first["reconstructions"]) == {"ego_self", "exo_self", "ego_swap", "exo_swap"}
    for name in first["reconstructions"]:
        torch.testing.assert_close(
            first["reconstructions"][name]["recon"],
            second["reconstructions"][name]["recon"],
            rtol=0,
            atol=0,
        )
