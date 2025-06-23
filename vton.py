#!/usr/bin/env python3
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from skimage.transform import PiecewiseAffineTransform, warp
import logging

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Exported symbols for use from other modules
__all__ = ["VTONPipeline", "process_vton", "virtual_try_on"]

# ========== rembg fallback ==========
try:
    from rembg import remove
except ImportError:
    logger.warning("rembg/onnxruntime not installed; background removal disabled.")
    def remove(img): return img

class VTONPipeline:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # DeepLabV3 for segmentation
        self.seg_model = models.segmentation.deeplabv3_resnet50(
            weights="DEFAULT"
        ).eval().to(self.device)
        logger.info("DeepLabV3 loaded.")

        # Keypoint R-CNN for human pose (nose, shoulders, hips, etc.)
        self.pose_model = models.detection.keypointrcnn_resnet50_fpn(
            pretrained=True
        ).eval().to(self.device)
        logger.info("Keypoint R-CNN loaded.")

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        )

    def segment(self, img: np.ndarray) -> np.ndarray:
        """Возвращает 0/255 маску переднего плана через DeepLabV3."""
        pil = Image.fromarray(img[...,::-1])  # BGR→RGB
        x = self.normalize(transforms.Resize((512,512))(self.to_tensor(pil)))
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.seg_model(x)["out"]
        mask = F.interpolate(out, size=(512,512), mode="bilinear", align_corners=False)
        mask = (mask.argmax(1).cpu().numpy()[0]>0).astype(np.uint8)*255
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    def extract_keypoints(self, img: np.ndarray):
        """
        Возвращает COCO-ключи: nose, left_shoulder, right_shoulder, left_hip, right_hip
        или None, если человек не найден.
        """
        # подготовка
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.to_tensor(rgb).to(self.device)
        with torch.no_grad():
            out = self.pose_model([tensor])[0]
        kps = out["keypoints"]        # [N, 17, 3]
        scores = out["keypoints_scores"] if "keypoints_scores" in out else None
        if kps.shape[0] == 0:
            return None
        # выбираем первое тело
        kp = kps[0].cpu().numpy()
        # индексы COCO keypoints
        # 0 - nose, 5 - left_shoulder, 6 - right_shoulder, 11 - left_hip, 12 - right_hip
        pts = {
            "nose":          tuple(kp[0,:2].astype(int)),
            "left_shoulder": tuple(kp[5,:2].astype(int)),
            "right_shoulder":tuple(kp[6,:2].astype(int)),
            "left_hip":      tuple(kp[11,:2].astype(int)),
            "right_hip":     tuple(kp[12,:2].astype(int)),
        }
        return pts

    def get_cloth_keypoints(self, mask: np.ndarray):
        """Ключевые точки одежды по bbox контура."""
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return {
            "nose":          (x + w//2, y),
            "left_shoulder": (x,       y + h//4),
            "right_shoulder":(x + w,   y + h//4),
            "left_hip":      (x,       y + 3*h//4),
            "right_hip":     (x + w,   y + 3*h//4),
        }

    def warp(self, cloth, mask, src_pts, dst_pts):
        """Piecewise-affine warp."""
        src = np.array([src_pts[k] for k in ["nose","left_shoulder","right_shoulder","left_hip","right_hip"]])
        dst = np.array([dst_pts[k] for k in ["nose","left_shoulder","right_shoulder","left_hip","right_hip"]])
        tfm = PiecewiseAffineTransform()
        if not tfm.estimate(src, dst):
            raise RuntimeError("Warp estimation failed")
        ci = cloth.astype(np.float32)/255.0
        cm = mask.astype(np.float32)/255.0
        wi = warp(ci, tfm, output_shape=ci.shape)
        wm = warp(cm, tfm, output_shape=cm.shape)
        return (wi*255).astype(np.uint8), (wm>0.5).astype(np.uint8)*255

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
        kp_cloth  = self.get_cloth_keypoints(mask_crop)
        if kp_person is None or kp_cloth is None:
            raise RuntimeError("Keypoint extraction failed")

        # 4) Warp + Blend
        warped_cloth, warped_mask = self.warp(cloth_crop, mask_crop, kp_cloth, kp_person)
        result = self.blend(person, warped_cloth, warped_mask)

        # 5) Сохраняем
        Image.fromarray(result).save(out_path)
        logger.info(f"Saved result to {out_path}")
        return out_path

def process_vton(person_path, cloth_path, output_path):
    return VTONPipeline().run(person_path, cloth_path, output_path)

def virtual_try_on(person_path, cloth_path):
    """Wrapper for easier use from external modules."""
    out_path = os.path.join(os.path.dirname(person_path), "vton_result.jpg")
    return process_vton(person_path, cloth_path, out_path)

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

    for p in (args.person, args.cloth):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            cv2.imwrite(p, np.zeros((1024, 1024, 3), dtype=np.uint8))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    process_vton(args.person, args.cloth, args.out)
