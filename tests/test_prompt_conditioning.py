import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from types import SimpleNamespace

from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
from sam_3d_body.data.transforms import TopdownAffine

class TestPromptConditioning(unittest.TestCase):
    def setUp(self):
        # Mock model and config
        self.mock_model = MagicMock()
        self.mock_model.device = "cpu"
        self.mock_model.head_pose.faces.cpu().numpy.return_value = np.zeros((10, 3))
        
        # Return a dummy list of outputs when run_inference is called
        dummy_out = {
            "mhr": {
                "focal_length": [torch.ones(1)],
                "pred_keypoints_3d": [torch.zeros(70, 3)],
                "pred_keypoints_2d": [torch.zeros(70, 2)],
                "pred_vertices": [torch.zeros(6890, 3)],
                "pred_cam_t": [torch.zeros(3)],
                "pred_pose_raw": [torch.zeros(72)],
                "global_rot": [torch.zeros(3)],
                "body_pose": [torch.zeros(69)],
                "hand": [torch.zeros(45)],
                "scale": [torch.zeros(10)],
                "shape": [torch.zeros(10)],
                "face": [torch.zeros(50)],
                "pred_joint_coords": [torch.zeros(70, 3)],
                "joint_global_rots": [torch.zeros(70, 3, 3)],
                "mhr_model_params": [{}],
            }
        }
        # mock returns a tuple of (pose_output, batch_lhand, batch_rhand, ...) for full inference
        # or just pose_output
        self.mock_model.run_inference.return_value = dummy_out
        
        self.mock_cfg = SimpleNamespace()
        self.mock_cfg.MODEL = SimpleNamespace()
        self.mock_cfg.MODEL.IMAGE_SIZE = [256, 192]
        
        self.estimator = SAM3DBodyEstimator(self.mock_model, self.mock_cfg)

    def test_keypoint_prompt_wiring(self):
        # Synthetic image [H, W, 3]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Fake bboxes
        bboxes = np.array([[10, 10, 100, 200]])
        
        # Fake keypoints [1, 70, 2] and scores [1, 70]
        # Make one keypoint high confidence and clearly inside bounds
        keypoints_2d = np.zeros((1, 70, 2), dtype=np.float32)
        keypoints_2d[0, 5] = [50, 100]  # Valid point inside bbox
        
        scores = np.zeros((1, 70), dtype=np.float32)
        scores[0, 5] = 0.9
        
        try:
            out = self.estimator.process_one_image(
                img=img,
                bboxes=bboxes,
                keypoints_2d=keypoints_2d,
                keypoint_scores=scores,
                keypoint_format="mhr70",
                inference_type="body"
            )
        except Exception as e:
            self.fail(f"process_one_image raised Exception unexpectedly: {e}")
            
        # Verify that _initialize_batch was called and check the batch argument
        self.assertTrue(self.mock_model._initialize_batch.called)
        
        batch = self.mock_model._initialize_batch.call_args[0][0]
        self.assertTrue(batch["has_keypoints"])
        
        # Check shape is [B, N, K, 3] (where N is 1 person usually but here just [1, K, 3] because batch size is 1)
        # Actually batch output is [1, 1, 70, 3] from prepare_batch? Wait, prepare_batch output is unsqueezed
        # Let's just check the shape of keypoints_2d in batch
        kps = batch["keypoints_2d"]
        self.assertEqual(kps.shape[-1], 3, f"Expected 3 channels (x, y, conf), got {kps.shape[-1]}")
        
        # Check normalization bounds for the valid keypoint
        # The 5th keypoint should have been normalized to [-0.5, 0.5]
        # and not masked out!
        val_x, val_y, conf = kps[0, 0, 5]
        self.assertAlmostEqual(conf.item(), 0.9, places=6, msg="Confidence score was lost")
        self.assertTrue(-0.5 <= val_x <= 0.5, f"X coord not normalized: {val_x}")
        self.assertTrue(-0.5 <= val_y <= 0.5, f"Y coord not normalized: {val_y}")

if __name__ == '__main__':
    unittest.main()
