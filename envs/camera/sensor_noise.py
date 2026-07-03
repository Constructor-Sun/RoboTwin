import cv2
import numpy as np


SENSOR_NOISE_TYPES = [
    "motion_blur",
    "gaussian_blur",
    "zoom_blur",
    "fog",
    "glass_blur",
]

SENSOR_NOISE_LABELS = {
    "motion_blur": "N1 motion blur",
    "gaussian_blur": "N2 gaussian blur",
    "zoom_blur": "N3 zoom blur",
    "fog": "N4 fog",
    "glass_blur": "N5 glass blur",
}


def _odd_kernel_size(value):
    value = max(3, int(round(value)))
    return value if value % 2 == 1 else value + 1


def _normalize_kernel(kernel):
    total = float(kernel.sum())
    if total <= 1e-8:
        return kernel
    return kernel / total


def _severity_ratio(severity):
    return np.clip((float(severity) - 1.0) / 4.0, 0.0, 1.0)


def _lerp(low, high, ratio):
    return float(low) + (float(high) - float(low)) * float(ratio)


class SensorNoiseProcessor:
    def __init__(self, config, episode_idx=0, rng=None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.cameras = set(self.config.get("cameras", ["head_camera", "left_camera", "right_camera"]))
        self.noise_types = list(self.config.get("types", SENSOR_NOISE_TYPES))
        if not self.noise_types:
            self.noise_types = list(SENSOR_NOISE_TYPES)

        self.episode_idx = int(episode_idx)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise_type = self._select_noise_type()
        self.severity = self._sample_severity()
        self.params = self._sample_params()
        self._glass_map_cache = {}

    def _select_noise_type(self):
        if self.config.get("cycle_by_episode", True):
            return self.noise_types[self.episode_idx % len(self.noise_types)]
        return self.noise_types[int(self.rng.integers(0, len(self.noise_types)))]

    def _sample_severity(self):
        severity_range = self.config.get("severity_range", [2, 3])
        return float(self.rng.uniform(float(severity_range[0]), float(severity_range[1])))

    def _sample_params(self):
        ratio = _severity_ratio(self.severity)

        if self.noise_type == "motion_blur":
            radius = _lerp(3.0, 15.0, ratio)
            sigma = _lerp(1.0, 8.0, ratio)
            angle = self.rng.uniform(-30.0, 30.0)
            return {
                "radius": radius,
                "sigma": sigma,
                "angle": angle,
                "kernel_size": _odd_kernel_size(radius),
            }

        if self.noise_type == "gaussian_blur":
            sigma = _lerp(1.0, 10.0, ratio)
            return {
                "sigma": sigma,
                "kernel_size": _odd_kernel_size(2 * int(3 * sigma) + 1),
            }

        if self.noise_type == "zoom_blur":
            max_scale = _lerp(1.0, 1.56, ratio)
            step = _lerp(0.03, 0.01, ratio)
            scales = np.arange(1.0, max_scale + step * 0.5, step, dtype=np.float32)
            return {"scales": scales}

        if self.noise_type == "fog":
            alpha = _lerp(0.3, 1.5, ratio)
            depth = float(self.config.get("fog_depth", 3.0))
            transmittance = np.exp(-alpha * depth)
            return {
                "alpha": alpha,
                "depth": depth,
                "transmittance": transmittance,
            }

        if self.noise_type == "glass_blur":
            displacement = _lerp(1.0, 5.0, ratio)
            sigma = _lerp(0.5, 2.5, ratio)
            iterations = int(round(_lerp(1, 3, ratio)))
            return {
                "displacement": displacement,
                "sigma": sigma,
                "iterations": iterations,
            }

        raise ValueError(f"Unsupported sensor noise type: {self.noise_type}")

    def should_apply(self, camera_name):
        return self.enabled and camera_name in self.cameras

    def summary(self):
        return {
            "enabled": self.enabled,
            "episode_idx": self.episode_idx,
            "noise_type": self.noise_type,
            "noise_label": SENSOR_NOISE_LABELS.get(self.noise_type, self.noise_type),
            "severity": self.severity,
            "cameras": sorted(self.cameras),
        }

    def apply(self, image, camera_name):
        if not self.should_apply(camera_name):
            return image

        if self.noise_type == "motion_blur":
            out = self._motion_blur(image)
        elif self.noise_type == "gaussian_blur":
            out = self._gaussian_blur(image)
        elif self.noise_type == "zoom_blur":
            out = self._zoom_blur(image)
        elif self.noise_type == "fog":
            out = self._fog(image)
        elif self.noise_type == "glass_blur":
            out = self._glass_blur(image, camera_name)
        else:
            raise ValueError(f"Unsupported sensor noise type: {self.noise_type}")

        return np.clip(out, 0, 255).astype(np.uint8)

    def _motion_blur(self, image):
        ksize = self.params["kernel_size"]
        sigma = self.params["sigma"]
        angle = self.params["angle"]

        coord = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
        kernel_1d = np.exp(-(coord ** 2) / (2.0 * sigma ** 2))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = kernel_1d

        center = ((ksize - 1) / 2.0, (ksize - 1) / 2.0)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (ksize, ksize))
        kernel = _normalize_kernel(kernel)
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    def _gaussian_blur(self, image):
        ksize = self.params["kernel_size"]
        sigma = self.params["sigma"]
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def _zoom_blur(self, image):
        h, w = image.shape[:2]
        accum = image.astype(np.float32)
        count = 1

        for scale in self.params["scales"][1:]:
            resized = cv2.resize(image, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            y0 = max(0, (rh - h) // 2)
            x0 = max(0, (rw - w) // 2)
            cropped = resized[y0:y0 + h, x0:x0 + w]
            if cropped.shape[:2] != (h, w):
                cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            accum += cropped.astype(np.float32)
            count += 1

        return accum / count

    def _fog(self, image):
        transmittance = self.params["transmittance"]
        white = np.full_like(image, 255, dtype=np.float32)
        return image.astype(np.float32) * transmittance + white * (1.0 - transmittance)

    def _get_glass_maps(self, image, camera_name):
        h, w = image.shape[:2]
        cache_key = (camera_name, h, w)
        if cache_key in self._glass_map_cache:
            return self._glass_map_cache[cache_key]

        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        maps = []
        for _ in range(self.params["iterations"]):
            dx = self.rng.uniform(-self.params["displacement"], self.params["displacement"], size=(h, w)).astype(np.float32)
            dy = self.rng.uniform(-self.params["displacement"], self.params["displacement"], size=(h, w)).astype(np.float32)
            dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=self.params["sigma"], sigmaY=self.params["sigma"])
            dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=self.params["sigma"], sigmaY=self.params["sigma"])
            maps.append((xx + dx, yy + dy))

        self._glass_map_cache[cache_key] = maps
        return maps

    def _glass_blur(self, image, camera_name):
        out = image.copy()
        for map_x, map_y in self._get_glass_maps(image, camera_name):
            out = cv2.remap(
                out,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=self.params["sigma"], sigmaY=self.params["sigma"])

        return out
