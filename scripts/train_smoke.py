"""Standalone 2-epoch dry-run smoke test script for VN-CoastSeg-Mask2Former."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mask2former import MultispectralMask2Former
from src.training.trainer import TwoStageTrainer


def run_smoke_test():
    print("=" * 64)
    print("VN-CoastSeg-Mask2Former: 2-Epoch Dry-Run Smoke Test")
    print("=" * 64)

    model = MultispectralMask2Former(
        in_channels=7,
        num_classes=6,
        num_queries=16,
        num_decoder_layers=2,
    )
    trainer = TwoStageTrainer(
        model=model,
        checkpoint_dir="checkpoints",
        lr_backbone=1e-5,
        lr_decoder=1e-4,
    )

    print("\nExecuting 2-epoch dry-run simulation...")
    result = trainer.dry_run(num_epochs=2, num_batches=2, batch_size=2)

    print("\n" + "=" * 64)
    print("DRY-RUN SMOKE TEST RESULTS")
    print("=" * 64)
    print(f"Epochs Completed : {result['epochs_completed']}")
    print(f"Initial Loss     : {result['initial_loss']:.4f}")
    print(f"Final Loss       : {result['final_loss']:.4f}")
    print(f"Loss Decreased   : {result['loss_decreased']}")
    print(f"Convergence      : {result['status']}")
    print(f"Checkpoint Saved : {result['checkpoint_path']}")
    print("=" * 64)

    assert result["loss_decreased"] is True
    print("[SUCCESS] All smoke test assertions verified successfully!")


if __name__ == "__main__":
    run_smoke_test()
