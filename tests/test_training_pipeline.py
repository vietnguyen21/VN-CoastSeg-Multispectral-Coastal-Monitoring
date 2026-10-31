"""Unit and integration test suite for TwoStageTrainer and Training Pipeline."""

import os
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    from src.models.torch_compat import torch
    HAS_TORCH = False

from src.models.mask2former import MultispectralMask2Former
from src.training.trainer import TwoStageTrainer


class TestTrainingPipeline(unittest.TestCase):
    """Test suite for TwoStageTrainer, differential optimization, and checkpointing."""

    def setUp(self):
        self.test_checkpoint_dir = Path("test_checkpoints_dir")
        self.test_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = MultispectralMask2Former(
            in_channels=7,
            num_classes=6,
            num_queries=16,
            num_decoder_layers=2,
        )
        self.trainer = TwoStageTrainer(
            model=self.model,
            checkpoint_dir=str(self.test_checkpoint_dir),
            lr_backbone=1e-5,
            lr_decoder=1e-4,
        )

    def tearDown(self):
        if self.test_checkpoint_dir.exists():
            shutil.rmtree(self.test_checkpoint_dir, ignore_errors=True)

    def test_stage_1_backbone_freezing(self):
        """Stage 1 must freeze all backbone parameters and train decoder with lr=1e-4."""
        self.trainer._build_optimizer(stage=1)

        # Check backbone parameter freezing
        for p in self.model.backbone.parameters():
            self.assertFalse(p.requires_grad)

        # Check decoder parameter trainability
        for p in self.model.transformer_decoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_stage_2_differential_learning_rates(self):
        """Stage 2 must unfreeze backbone and configure differential learning rates."""
        optimizer = self.trainer._build_optimizer(stage=2)

        # All parameters must be trainable
        for p in self.model.parameters():
            self.assertTrue(p.requires_grad)

        # 2 parameter groups: backbone lr = 1e-5, decoder lr = 1e-4
        self.assertEqual(len(optimizer.param_groups), 2)
        bb_group = optimizer.param_groups[0]
        dec_group = optimizer.param_groups[1]

        self.assertAlmostEqual(bb_group["lr"], 1e-5)
        self.assertAlmostEqual(dec_group["lr"], 1e-4)

    def test_checkpoint_save_and_restore(self):
        """Saved checkpoint must accurately serialize and restore model state."""
        ckpt_path = self.trainer.save_checkpoint(
            epoch=5,
            metrics={"mIoU": 0.725},
            filename="test_ckpt_epoch_5.pt",
        )
        self.assertTrue(os.path.exists(ckpt_path))

        # Fresh model instance
        fresh_model = MultispectralMask2Former(
            in_channels=7,
            num_classes=6,
            num_queries=16,
            num_decoder_layers=2,
        )
        fresh_trainer = TwoStageTrainer(
            model=fresh_model,
            checkpoint_dir=str(self.test_checkpoint_dir),
        )
        fresh_trainer.load_checkpoint(ckpt_path)
        self.assertEqual(fresh_trainer.epoch, 5)

    def test_dry_run_smoke_test_convergence(self):
        """Dry-run smoke test (2 epochs) must complete, verify decreasing loss, and save checkpoint."""
        summary = self.trainer.dry_run(num_epochs=2, num_batches=2, batch_size=2)

        self.assertEqual(summary["epochs_completed"], 2)
        self.assertTrue(summary["loss_decreased"])
        self.assertEqual(summary["status"], "CONVERGED")
        self.assertTrue(os.path.exists(summary["checkpoint_path"]))


if __name__ == "__main__":
    unittest.main()
