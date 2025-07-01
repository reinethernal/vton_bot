#!/usr/bin/env python3
import os
# Suppress verbose logs from TensorFlow and Mediapipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging._warn_preinit_stderr = False

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms, models
except ImportError:  # pragma: no cover - optional heavy deps
    torch = None
    F = None
    transforms = None
    models = None
from skimage.transform import PiecewiseAffineTransform, warp
import logging

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Exported symbols for use from other modules
__all__ = [
    "VTONPipeline",
    "process_vton",
    "virtual_try_on",
    "get_pipeline",
]

# ========== rembg fallback ==========
try:
    from rembg import remove
except ImportError:
    logger.warning("rembg/onnxruntime not installed; background removal disabled.")
    def remove(img): return img

class VTONPipeline:
    def __init__(self, segmentation_model: str = "deeplabv3"):
        if torch is None or transforms is None or models is None:
            raise RuntimeError("PyTorch and torchvision are required for VTONPipeline")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Segmentation model
        self.seg_model_name = segmentation_model
        if segmentation_model == "deeplabv3":
            self.seg_model = models.segmentation.deeplabv3_resnet50(
                weights="DEFAULT"
            ).eval().to(self.device)
            logger.info("DeepLabV3 loaded.")
        elif segmentation_model == "u2net":
            from u2net import U2NET

            weights = Path("models/cloth_segm_u2net_latest.pth")
            if not weights.exists():
                raise FileNotFoundError(f"U2Net weights not found at {weights}")
            self.seg_model = U2NET(3, 1)
            self.seg_model.load_state_dict(
                torch.load(weights, map_location=self.device)
            )
            self.seg_model.eval().to(self.device)
            logger.info("U2Net loaded from %s", weights)
        else:
            raise ValueError(f"Unknown segmentation model: {segmentation_model}")

        # Initialize OpenPose BODY_25 pose estimation unconditionally if the
        # Python bindings are available. The OpenPose models are expected to be
        # located under ``openpose/models`` relative to this file. If the
        # bindings cannot be imported, we fall back to Mediapipe.
        try:  # pragma: no cover - optional heavy deps
            from openpose import pyopenpose as op  # type: ignore

            model_dir = os.environ.get("OPENPOSE_MODEL_DIR")
            if model_dir is None:
                model_dir = str(Path(__file__).resolve().parent / "openpose" / "models")
            params = {
                "model_folder": model_dir,
                "model_pose": "BODY_25",
            }
            self.op = op
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()
            self.pose_backend = "openpose"
            logger.info("OpenPose успешно загружен из %s", params["model_folder"])

            # Initialize Mediapipe as a fallback if available
            try:
                import mediapipe as mp

                self.mp = mp
                self.mp_pose = self.mp.solutions.pose.Pose(static_image_mode=True)
            except ImportError:  # pragma: no cover - optional dep
                self.mp = None
                self.mp_pose = None
        except ImportError:
            import mediapipe as mp

            self.mp = mp
            self.mp_pose = self.mp.solutions.pose.Pose(static_image_mode=True)
            self.pose_backend = "mediapipe"
            self.op = None
            self.op_wrapper = None
            logger.warning("OpenPose не доступен; перехожу на Mediapipe")

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        )

    def segment(self, img: np.ndarray) -> np.ndarray:
        """Return a 0/255 garment mask."""
        pil = Image.fromarray(img[..., ::-1])  # BGR→RGB
        x = self.normalize(transforms.Resize((512, 512))(self.to_tensor(pil)))
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.seg_model_name == "deeplabv3":
                out = self.seg_model(x)["out"]
                mask = F.interpolate(
                    out, size=(512, 512), mode="bilinear", align_corners=False
                )
                mask = mask.argmax(1)
            else:  # u2net
                mask, *_ = self.seg_model(x)
        mask = mask.squeeze(0)
        mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def extract_keypoints(self, img: np.ndarray):
        """Return keypoints for the main person in the image."""
        if self.pose_backend == "openpose":  # pragma: no cover - heavy path
            datum = self.op.Datum()  # type: ignore[attr-defined]
            datum.cvInputData = img
            # Explicitly wrap the datum list using VectorDatum for
            # compatibility with builds lacking automatic STL bindings.
            try:
                datums = self.op.VectorDatum([datum])
            except AttributeError:
                datums = [datum]
            self.op_wrapper.emplaceAndPop(datums)
            pts_arr = datum.poseKeypoints
            if pts_arr is None or pts_arr.size == 0:
                if self.mp_pose is not None:
                    backend_name = (
                        "OpenPose" if self.pose_backend == "openpose" else self.pose_backend
                    )
                    logger.warning(
                        f"{backend_name} detected no keypoints on {img.shape}; switching to Mediapipe"
                    )
                    return self._extract_keypoints_mp(img)
                return None
            kp = pts_arr[0]
            mapping = {
                "nose": 0,
                "neck": 1,
                "right_shoulder": 2,
                "right_elbow": 3,
                "right_wrist": 4,
                "left_shoulder": 5,
                "left_elbow": 6,
                "left_wrist": 7,
                "mid_hip": 8,
                "right_hip": 9,
                "right_knee": 10,
                "right_ankle": 11,
                "left_hip": 12,
                "left_knee": 13,
                "left_ankle": 14,
                "right_eye": 15,
                "left_eye": 16,
                "right_ear": 17,
                "left_ear": 18,
                "left_big_toe": 19,
                "left_small_toe": 20,
                "left_heel": 21,
                "right_big_toe": 22,
                "right_small_toe": 23,
                "right_heel": 24,
            }
            pts = {k: tuple(kp[i, :2].astype(int)) for k, i in mapping.items()}
            return pts
        else:
            return self._extract_keypoints_mp(img)

    def _extract_keypoints_mp(self, img: np.ndarray):
        """Mediapipe-based keypoint extraction."""
        # Mediapipe expects images in RGB order. Convert from BGR before
        # constructing the ``mp.Image`` instance to avoid orientation issues.
        rgb = img[..., ::-1].copy()

        # ``self.mp_pose.process`` changed in Mediapipe 0.10 to accept
        # ``mp.Image`` instead of ``np.ndarray``. Try the new API first and
        # fall back to the old behavior if necessary.
        try:
            mp_image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB, data=rgb
            )
            results = self.mp_pose.process(mp_image)
        except AttributeError as e:
            if "shape" not in str(e):
                raise
            results = self.mp_pose.process(rgb)
        lm = results.pose_landmarks
        if not lm:
            return None
        h, w = img.shape[:2]
        idx = self.mp.solutions.pose.PoseLandmark  # type: ignore[attr-defined]
        mapping = {
            "nose": idx.NOSE,
            "left_eye": idx.LEFT_EYE,
            "right_eye": idx.RIGHT_EYE,
            "left_ear": idx.LEFT_EAR,
            "right_ear": idx.RIGHT_EAR,
            "left_shoulder": idx.LEFT_SHOULDER,
            "right_shoulder": idx.RIGHT_SHOULDER,
            "left_elbow": idx.LEFT_ELBOW,
            "right_elbow": idx.RIGHT_ELBOW,
            "left_wrist": idx.LEFT_WRIST,
            "right_wrist": idx.RIGHT_WRIST,
            "left_hip": idx.LEFT_HIP,
            "right_hip": idx.RIGHT_HIP,
            "left_knee": idx.LEFT_KNEE,
            "right_knee": idx.RIGHT_KNEE,
            "left_ankle": idx.LEFT_ANKLE,
            "right_ankle": idx.RIGHT_ANKLE,
            "left_heel": idx.LEFT_HEEL,
            "right_heel": idx.RIGHT_HEEL,
            "left_big_toe": idx.LEFT_FOOT_INDEX,
            "right_big_toe": idx.RIGHT_FOOT_INDEX,
        }
        pts = {}
        for name, landmark_idx in mapping.items():
            lm_pt = lm.landmark[landmark_idx]
            if lm_pt.visibility < 0.5:
                return None
            pts[name] = (int(lm_pt.x * w), int(lm_pt.y * h))
        # Derived points not directly provided by mediapipe
        pts["left_small_toe"] = pts["left_big_toe"]
        pts["right_small_toe"] = pts["right_big_toe"]
        pts["neck"] = (
            int((pts["left_shoulder"][0] + pts["right_shoulder"][0]) / 2),
            int((pts["left_shoulder"][1] + pts["right_shoulder"][1]) / 2),
        )
        pts["mid_hip"] = (
            int((pts["left_hip"][0] + pts["right_hip"][0]) / 2),
            int((pts["left_hip"][1] + pts["right_hip"][1]) / 2),
        )
        return pts

    def get_cloth_keypoints(self, cloth: np.ndarray, mask: np.ndarray):
        """Return cloth keypoints using OpenPose with bbox fallback.

        ``cloth`` is the cropped garment image and ``mask`` its segmentation
        mask. If OpenPose is available and manages to detect a body from the
        masked cloth image, those keypoints are returned. Otherwise we fall back
        to the previous heuristic based on the bounding box.
        """

        if self.pose_backend == "openpose":  # pragma: no cover - optional path
            seg = cv2.bitwise_and(cloth, cloth, mask=mask)
            pts = self.extract_keypoints(seg)
            if pts:
                return pts

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))

        def pt(px, py):
            return (
                int(np.clip(px, 0, mask.shape[1] - 1)),
                int(np.clip(py, 0, mask.shape[0] - 1)),
            )

        return {
            "nose": pt(x + w * 0.5, y),
            "neck": pt(x + w * 0.5, y + h * 0.2),
            "right_shoulder": pt(x + w * 0.9, y + h * 0.25),
            "right_elbow": pt(x + w * 0.95, y + h * 0.5),
            "right_wrist": pt(x + w, y + h * 0.75),
            "left_shoulder": pt(x + w * 0.1, y + h * 0.25),
            "left_elbow": pt(x + w * 0.05, y + h * 0.5),
            "left_wrist": pt(x, y + h * 0.75),
            "mid_hip": pt(x + w * 0.5, y + h * 0.9),
            "right_hip": pt(x + w * 0.75, y + h),
            "right_knee": pt(x + w * 0.75, y + h * 1.25),
            "right_ankle": pt(x + w * 0.75, y + h * 1.5),
            "left_hip": pt(x + w * 0.25, y + h),
            "left_knee": pt(x + w * 0.25, y + h * 1.25),
            "left_ankle": pt(x + w * 0.25, y + h * 1.5),
            "right_eye": pt(x + w * 0.55, y - h * 0.05),
            "left_eye": pt(x + w * 0.45, y - h * 0.05),
            "right_ear": pt(x + w * 0.65, y + h * 0.05),
            "left_ear": pt(x + w * 0.35, y + h * 0.05),
            "left_big_toe": pt(x + w * 0.3, y + h * 1.7),
            "left_small_toe": pt(x + w * 0.2, y + h * 1.7),
            "left_heel": pt(x + w * 0.25, y + h * 1.5),
            "right_big_toe": pt(x + w * 0.8, y + h * 1.7),
            "right_small_toe": pt(x + w * 0.7, y + h * 1.7),
            "right_heel": pt(x + w * 0.75, y + h * 1.5),
        }

    def warp(self, cloth, mask, src_pts, dst_pts, person_shape=None, *, return_status=False):
        """Piecewise-affine warp with fallback to simple scaling.

        If ``return_status`` is ``True`` an extra string is returned
        indicating whether the warp succeeded (``"ok"``), failed during
        estimation (``"estimation_failed"``) or another error occurred
        (``"error"``).
        """
        keys = [
            "nose",
            "neck",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "mid_hip",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_hip",
            "left_knee",
            "left_ankle",
            "right_eye",
            "left_eye",
            "right_ear",
            "left_ear",
            "left_big_toe",
            "left_small_toe",
            "left_heel",
            "right_big_toe",
            "right_small_toe",
            "right_heel",
        ]
        src = np.array([src_pts[k] for k in keys if k in src_pts], dtype=np.float32)
        dst = np.array([dst_pts[k] for k in keys if k in dst_pts], dtype=np.float32)
        if len(src) < 3 or len(dst) < 3:
            raise RuntimeError("Not enough keypoints")
        tfm = PiecewiseAffineTransform()

        def approx_overlay():
            if person_shape is None:
                return cloth, mask
            ph, pw = person_shape
            bbox_pts = list(dst_pts.values())
            x0 = min(p[0] for p in bbox_pts)
            x1 = max(p[0] for p in bbox_pts)
            y0 = min(p[1] for p in bbox_pts)
            y1 = max(p[1] for p in bbox_pts)

            # map bbox to cloth coordinate system
            cx0 = int(np.clip(x0 / pw * cloth.shape[1], 0, cloth.shape[1] - 1))
            cx1 = int(np.clip(x1 / pw * cloth.shape[1], cx0 + 1, cloth.shape[1]))
            cy0 = int(np.clip(y0 / ph * cloth.shape[0], 0, cloth.shape[0] - 1))
            cy1 = int(np.clip(y1 / ph * cloth.shape[0], cy0 + 1, cloth.shape[0]))

            new_w = max(cx1 - cx0, 1)
            new_h = max(cy1 - cy0, 1)
            scaled_cloth = cv2.resize(cloth, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            out_cloth = np.zeros_like(cloth)
            out_mask = np.zeros_like(mask)
            out_cloth[cy0:cy1, cx0:cx1] = scaled_cloth
            out_mask[cy0:cy1, cx0:cx1] = scaled_mask
            return out_cloth, out_mask

        status = "ok"
        try:
            if not tfm.estimate(src, dst):
                logger.warning(
                    "Warp estimation failed with %d source and %d destination points. Using approximate overlay.",
                    len(src),
                    len(dst),
                )
                status = "estimation_failed"
                result = approx_overlay()
                return (*result, status) if return_status else result

            fwd_pts = tfm(src)
            inv_pts = tfm.inverse(dst)
            if not (np.all(np.isfinite(fwd_pts)) and np.all(np.isfinite(inv_pts))):
                logger.warning("Warp transform invalid. Using approximate overlay.")
                status = "error"
                result = approx_overlay()
                return (*result, status) if return_status else result

            fwd_err = np.linalg.norm(fwd_pts - dst, axis=1)
            inv_err = np.linalg.norm(inv_pts - src, axis=1)
            if np.any(fwd_err > 10) or np.any(inv_err > 10):
                logger.warning("Warp error too large. Using approximate overlay.")
                status = "error"
                result = approx_overlay()
                return (*result, status) if return_status else result

            ci = cloth.astype(np.float32)/255.0
            cm = mask.astype(np.float32)/255.0
            wi = warp(ci, tfm, output_shape=ci.shape)
            wm = warp(cm, tfm, output_shape=cm.shape)
            result = ((wi*255).astype(np.uint8), (wm>0.5).astype(np.uint8)*255)
            return (*result, status) if return_status else result
        except Exception as e:
            logger.warning("Warp failed (%s). Using approximate overlay.", e)
            status = "error"
            result = approx_overlay()
            return (*result, status) if return_status else result

    def blend(self, person, cloth, mask):
        """Простое alpha-blend наложение."""
        h,w = person.shape[:2]
        c = cv2.resize(cloth, (w,h), interpolation=cv2.INTER_LANCZOS4)
        m = cv2.resize(mask,  (w,h), interpolation=cv2.INTER_NEAREST)
        alpha = (m.astype(np.float32)/255.0)[...,None]
        blended = (c.astype(np.float32)*alpha + person.astype(np.float32)*(1-alpha)).astype(np.uint8)
        return np.where(m[...,None]==255, blended, person)

    def run(self, person_path, cloth_path, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 1) Загрузка + удаление фона
        person = cv2.imread(person_path, cv2.IMREAD_UNCHANGED)
        if person is None:
            raise RuntimeError(f"Failed to load person image from '{person_path}'")
        person = np.array(remove(person))[:, :, :3]  # RGBA→RGB

        cloth = cv2.imread(cloth_path, cv2.IMREAD_COLOR)
        if cloth is None:
            raise RuntimeError(f"Failed to load cloth image from '{cloth_path}'")

        # 2) Сегментация одежды + кроп
        cloth_mask = self.segment(cloth)
        cnts,_ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Cloth segmentation failed")
        x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cloth_crop = cv2.resize(cloth[y:y+h, x:x+w], (512,512), interpolation=cv2.INTER_LANCZOS4)
        mask_crop  = cv2.resize(cloth_mask[y:y+h, x:x+w], (512,512), interpolation=cv2.INTER_NEAREST)

        # 3) Ключевые точки
        kp_person = self.extract_keypoints(person)
        kp_cloth  = self.get_cloth_keypoints(cloth_crop, mask_crop)
        if kp_person is None or kp_cloth is None:
            raise RuntimeError("Keypoint extraction failed")

        # 4) Warp + Blend
        warped_cloth, warped_mask = self.warp(
            cloth_crop, mask_crop, kp_cloth, kp_person, person.shape[:2]
        )
        result = self.blend(person, warped_cloth, warped_mask)

        # 5) Сохраняем
        Image.fromarray(result).save(out_path)
        logger.info(f"Saved result to {out_path}")
        return out_path


_GLOBAL_PIPELINE = None


def get_pipeline():
    """Return a cached :class:`VTONPipeline` instance."""
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = VTONPipeline()
    return _GLOBAL_PIPELINE

def process_vton(person_path, cloth_path, output_path, pipeline=None):
    """Run the virtual try-on pipeline."""
    if pipeline is None:
        pipeline = get_pipeline()
    return pipeline.run(person_path, cloth_path, output_path)

def virtual_try_on(person_path, cloth_path, pipeline=None):
    """Convenience wrapper returning the output path of the result image."""
    out_path = os.path.join(os.path.dirname(person_path), "vton_result.jpg")
    return process_vton(person_path, cloth_path, out_path, pipeline=pipeline)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run a simple VTON example")
    cwd = Path.cwd()
    parser.add_argument(
        "--person",
        default=str(cwd / "temp_person.jpg"),
        help="Path to the person image",
    )
    parser.add_argument(
        "--cloth",
        default=str(cwd / "static" / "uniforms" / "uniform1.png"),
        help="Path to the cloth image",
    )
    parser.add_argument(
        "--out",
        default=str(cwd / "tmp" / "vton_result.jpg"),
        help="Where to save the result image",
    )
    args = parser.parse_args()

    for path, opt in ((args.person, "--person"), (args.cloth, "--cloth")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File specified via {opt} does not exist: '{path}'")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    process_vton(args.person, args.cloth, args.out)
