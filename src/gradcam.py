import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        activations = {}
        gradients = {}

        def fwd_hook(_, __, output):
            activations["value"] = output.detach()

        def bwd_hook(_, __, grad_output):
            gradients["value"] = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(fwd_hook)
        h_bwd = self.target_layer.register_full_backward_hook(bwd_hook)

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            with torch.enable_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

                if class_idx is None:
                    class_idx = int(torch.argmax(logits, dim=1).item())

                score = logits[0, class_idx]
                score.backward()

            acts = activations["value"]          # [1, C, H, W]
            grads = gradients["value"]           # [1, C, H, W]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1).squeeze(0)
            cam = torch.relu(cam).cpu().numpy()

            cam_max = cam.max()
            if cam_max > 0:
                cam = (cam - cam.min()) / (cam_max - cam.min() + 1e-8)

            h, w = input_tensor.shape[2], input_tensor.shape[3]
            cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
            cam_resized = np.clip(cam_resized, 0.0, 1.0)
            return cam_resized, probs, class_idx
        finally:
            h_fwd.remove()
            h_bwd.remove()


def overlay_heatmap(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blends Grad-CAM onto a 2D grayscale X-ray and returns an RGB uint8 image."""
    base_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return blended