"""
🚀 BANNER RESIZER ULTRA LITE - Professional Banner Generator
Không cần AI models - Chạy ngay lập tức
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import warnings
import time
import os
import sys
import json
import io
import zipfile
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import hashlib
import traceback

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Cấu hình hệ thống"""
    MAX_IMAGE_SIZE = 4096
    DEVICE = "cpu"
    
# ==================== SMART IMAGE ANALYSIS ====================
class ImageAnalyzer:
    """Phân tích ảnh thông minh không cần AI"""
    
    @staticmethod
    def analyze_content(image: Image.Image) -> Dict[str, Any]:
        """Phân tích nội dung ảnh bằng computer vision"""
        try:
            img_np = np.array(image)
            
            # Chuyển sang grayscale để phân tích
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            # Phát hiện edges
            edges = cv2.Canny(gray, 100, 200)
            
            # Phát hiện faces (nếu có OpenCV với face detection)
            has_faces = False
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                has_faces = len(faces) > 0
            except:
                pass
            
            # Phân tích màu sắc
            dominant_colors = ImageAnalyzer._extract_dominant_colors(image)
            
            # Phân tích bố cục
            composition = ImageAnalyzer._analyze_composition(gray, edges)
            
            # Tạo mô tả thông minh
            description = ImageAnalyzer._generate_description(img_np, has_faces, composition)
            
            # Tạo tags
            tags = []
            if has_faces:
                tags.append("people")
            if composition["edge_density"] > 0.1:
                tags.append("detailed")
            if len(dominant_colors) > 0 and max(dominant_colors[0]) > 200:
                tags.append("bright")
            
            return {
                "description": description,
                "tags": tags,
                "dominant_colors": dominant_colors,
                "composition": composition,
                "has_people": has_faces,
                "size": image.size,
                "format": image.format if hasattr(image, 'format') else 'RGB',
            }
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return {
                "description": "professional image",
                "tags": ["professional"],
                "dominant_colors": [],
                "composition": {},
                "has_people": False,
                "size": image.size,
            }
    
    @staticmethod
    def _extract_dominant_colors(image: Image.Image, n_colors: int = 5):
        """Trích xuất màu chủ đạo với k-means đơn giản"""
        try:
            # Resize ảnh để tăng tốc
            small_img = image.resize((100, 100))
            img_np = np.array(small_img)
            
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np] * 3, axis=2)
            
            pixels = img_np.reshape(-1, 3)
            
            # Lấy mẫu ngẫu nhiên
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]
            
            # Manual k-means đơn giản
            colors = []
            for _ in range(n_colors):
                if len(pixels) == 0:
                    break
                # Chọn một điểm ngẫu nhiên làm center
                center = pixels[np.random.randint(len(pixels))]
                colors.append(center.tolist())
            
            return colors
        except:
            return [[255, 255, 255], [200, 200, 200], [150, 150, 150]]
    
    @staticmethod
    def _analyze_composition(gray, edges):
        """Phân tích bố cục"""
        h, w = gray.shape
        
        # Tính edge density
        edge_density = np.mean(edges > 0) if edges.size > 0 else 0
        
        # Tính rule of thirds
        third_h, third_w = h // 3, w // 3
        intersections = [
            (third_w, third_h),
            (2 * third_w, third_h),
            (third_w, 2 * third_h),
            (2 * third_w, 2 * third_h)
        ]
        
        densities = []
        for x, y in intersections:
            if 0 <= y < h and 0 <= x < w:
                densities.append(edges[y, x] > 0)
        
        rule_of_thirds_score = np.mean(densities) if densities else 0
        
        # Tính center of mass từ edges
        y_indices, x_indices = np.where(edges > 0)
        if len(x_indices) > 0:
            center_x = np.mean(x_indices) / w
            center_y = np.mean(y_indices) / h
        else:
            center_x, center_y = 0.5, 0.5
        
        return {
            "edge_density": float(edge_density),
            "rule_of_thirds_score": float(rule_of_thirds_score),
            "center_of_mass": (float(center_x), float(center_y)),
            "contrast": float(np.std(gray)),
        }
    
    @staticmethod
    def _generate_description(img_np, has_faces, composition):
        """Tạo mô tả thông minh"""
        h, w = img_np.shape[:2]
        aspect_ratio = w / h
        
        descriptions = []
        
        # Dựa trên aspect ratio
        if aspect_ratio > 1.5:
            descriptions.append("landscape")
        elif aspect_ratio < 0.7:
            descriptions.append("portrait")
        else:
            descriptions.append("square format")
        
        # Dựa trên màu sắc
        if len(img_np.shape) == 3:
            avg_color = np.mean(img_np, axis=(0, 1))
            brightness = np.mean(avg_color)
            if brightness > 200:
                descriptions.append("bright")
            elif brightness < 100:
                descriptions.append("dark")
        
        # Dựa trên composition
        if composition["edge_density"] > 0.15:
            descriptions.append("detailed")
        if composition["rule_of_thirds_score"] > 0.5:
            descriptions.append("well-composed")
        
        if has_faces:
            descriptions.append("with people")
        
        return " ".join(descriptions) + " image"

# ==================== GENERATIVE EXPAND ENGINE ====================
class GenerativeExpandEngine:
    """Mở rộng ảnh thông minh không cần AI"""
    
    def expand_image(self, image: Image.Image, target_size: Tuple[int, int], 
                    context_prompt: str = "") -> Image.Image:
        """
        Mở rộng ảnh với smart content-aware fill (không cần AI)

        ✅ Fix stability:
        - Nếu ảnh gốc lớn hơn target ở *bất kỳ* chiều nào -> auto scale down để FIT vào canvas
          (tránh lỗi out-of-bounds khi paste ảnh cao hơn target_h).
        - Sau đó mới tiến hành expand phần thiếu.
        """
        image = image.convert("RGB")
        w, h = image.size
        target_w, target_h = target_size

        # Nếu target nhỏ hơn hoặc bằng ảnh gốc ở cả 2 chiều -> resize bình thường
        if target_w <= w and target_h <= h:
            return image.resize(target_size, Image.Resampling.LANCZOS)

        # Nếu ảnh gốc lớn hơn target ở một trong hai chiều -> scale down để FIT
        # (tránh paste ảnh to hơn canvas gây lỗi index trong bước fill/blend)
        if w > target_w or h > target_h:
            scale = min(target_w / w, target_h / h)
            scale = max(0.01, min(1.0, scale))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = image.size

        # Tạo canvas mới
        canvas = Image.new("RGB", target_size, (255, 255, 255))

        # Đặt ảnh gốc vào giữa
        x_offset = max(0, (target_w - w) // 2)
        y_offset = max(0, (target_h - h) // 2)

        canvas.paste(image, (x_offset, y_offset))

        # Smart fill vùng trống
        result = self._smart_fill_expansion(canvas, x_offset, y_offset, w, h)

        return result

    def _smart_fill_expansion(self, canvas, x_offset, y_offset, w, h):
        """Điền vùng trống thông minh (OpenCV inpaint + blend, đã fix out-of-bounds)."""
        img_np = np.array(canvas)
        target_h, target_w = img_np.shape[:2]

        # Clamp vùng content cho an toàn (tránh paste ảnh lớn hơn canvas gây lỗi index)
        x1 = max(0, int(x_offset))
        y1 = max(0, int(y_offset))
        x2 = min(target_w, int(x_offset + w))
        y2 = min(target_h, int(y_offset + h))

        # Nếu content box rỗng -> trả luôn canvas
        if x2 <= x1 or y2 <= y1:
            return canvas

        # Tạo mask: vùng content = 255, vùng cần inpaint = 0 (sau đó invert để inpaint phần trống)
        mask = np.zeros((target_h, target_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        try:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Inpaint vùng trống (255 - mask)
            kernel = np.ones((3, 3), np.uint8)
            mask_dilated = cv2.dilate(255 - mask, kernel, iterations=3)

            inpainted = cv2.inpaint(img_bgr, mask_dilated, 3, cv2.INPAINT_TELEA)
            result_np = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

            # Blend nhẹ vùng biên giữa content gốc và vùng inpaint để giảm seam
            blend_mask = np.ones((target_h, target_w, 3), dtype=np.float32)

            # blend_width tùy theo kích thước content box (tránh quá lớn với ảnh nhỏ)
            cw = x2 - x1
            ch = y2 - y1
            blend_width = int(min(20, max(6, min(cw, ch) * 0.08)))

            for i in range(blend_width):
                alpha = i / max(1, (blend_width - 1))  # 0..1

                # Top edge inside content
                yt = y1 + i
                if 0 <= yt < target_h:
                    blend_mask[yt, x1:x2] = alpha

                # Bottom edge inside content
                yb = y2 - i - 1
                if 0 <= yb < target_h:
                    blend_mask[yb, x1:x2] = alpha

                # Left edge inside content
                xl = x1 + i
                if 0 <= xl < target_w:
                    blend_mask[y1:y2, xl] = alpha

                # Right edge inside content
                xr = x2 - i - 1
                if 0 <= xr < target_w:
                    blend_mask[y1:y2, xr] = alpha

            original_region = img_np[y1:y2, x1:x2]
            result_region = result_np[y1:y2, x1:x2]

            if original_region.shape == result_region.shape:
                blend_region = blend_mask[y1:y2, x1:x2]
                blended = (original_region * blend_region + result_region * (1 - blend_region)).astype(np.uint8)
                result_np[y1:y2, x1:x2] = blended

            return Image.fromarray(result_np)

        except Exception as e:
            print(f"⚠️ Inpainting failed: {e}")
            return self._mirror_padding_fallback(canvas, x_offset, y_offset, w, h)

def _mirror_padding_fallback(self, canvas, x_offset, y_offset, w, h):
        """Fallback với mirror padding (ổn định, không out-of-bounds)."""
        img_np = np.array(canvas)
        target_h, target_w = img_np.shape[:2]

        x1 = max(0, int(x_offset))
        y1 = max(0, int(y_offset))
        x2 = min(target_w, int(x_offset + w))
        y2 = min(target_h, int(y_offset + h))

        if x2 <= x1 or y2 <= y1:
            return canvas

        def _tile_vertical(src: np.ndarray, out_h: int) -> np.ndarray:
            if src.shape[0] == 0:
                return src
            reps = int(np.ceil(out_h / src.shape[0]))
            tiled = np.tile(src, (reps, 1, 1))
            return tiled[:out_h, :, :]

        def _tile_horizontal(src: np.ndarray, out_w: int) -> np.ndarray:
            if src.shape[1] == 0:
                return src
            reps = int(np.ceil(out_w / src.shape[1]))
            tiled = np.tile(src, (1, reps, 1))
            return tiled[:, :out_w, :]

        # Top fill
        if y1 > 0:
            height_to_fill = y1
            src = img_np[y1:min(y2, y1 + height_to_fill), x1:x2]
            if src.size > 0:
                fill = _tile_vertical(src[::-1], height_to_fill)
                img_np[:y1, x1:x2] = fill

        # Bottom fill
        if y2 < target_h:
            height_to_fill = target_h - y2
            src = img_np[max(y1, y2 - height_to_fill):y2, x1:x2]
            if src.size > 0:
                fill = _tile_vertical(src[::-1], height_to_fill)
                img_np[y2:target_h, x1:x2] = fill

        # Left fill
        if x1 > 0:
            width_to_fill = x1
            src = img_np[y1:y2, x1:min(x2, x1 + width_to_fill)]
            if src.size > 0:
                fill = _tile_horizontal(src[:, ::-1], width_to_fill)
                img_np[y1:y2, :x1] = fill

        # Right fill
        if x2 < target_w:
            width_to_fill = target_w - x2
            src = img_np[y1:y2, max(x1, x2 - width_to_fill):x2]
            if src.size > 0:
                fill = _tile_horizontal(src[:, ::-1], width_to_fill)
                img_np[y1:y2, x2:target_w] = fill

        # Fill corners (solid color from nearest pixel)
        img_np[:y1, :x1] = img_np[y1, x1]
        img_np[:y1, x2:target_w] = img_np[y1, x2 - 1]
        img_np[y2:target_h, :x1] = img_np[y2 - 1, x1]
        img_np[y2:target_h, x2:target_w] = img_np[y2 - 1, x2 - 1]

        return Image.fromarray(img_np)

# ==================== CELTRA REFLOW ENGINE ====================
class CeltraReflowEngine:
    """Celtra/Bannerflow-like reflow (No-AI, CPU-only).

    Mục tiêu:
    - Chuyển landscape banner -> portrait story / vertical placement
    - Giữ header (thường là text/top), giữ footer (mascot/scene/bottom)
    - Kéo giãn/đổ nền phần "sky/background" ở giữa để đủ chiều cao
    """

    def __init__(self):
        pass

    def reflow(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        image = image.convert("RGB")
        target_w, target_h = target_size
        src_w, src_h = image.size

        # Scale để vừa chiều rộng target
        scale = target_w / max(1, src_w)
        scaled_h = max(1, int(round(src_h * scale)))
        scaled = image.resize((target_w, scaled_h), Image.Resampling.LANCZOS)

        if scaled_h == target_h:
            return scaled

        if scaled_h > target_h:
            return self._smart_crop_preserve_top_bottom(scaled, target_h)

        return self._extend_middle_band(scaled, target_h)

    def _smart_crop_preserve_top_bottom(self, scaled: Image.Image, target_h: int) -> Image.Image:
        """Nếu scaled cao hơn target_h thì crop thông minh để giữ phần top/bottom."""
        w, h = scaled.size

        # Các mốc tương đối (tuning cho banner kiểu game/campaign)
        header_end = int(h * 0.28)
        footer_start = int(h * 0.62)

        # Muốn header_end nằm gần 28% chiều cao target, footer_start gần ~75%
        desired_header_y = int(target_h * 0.28)
        crop_y = max(0, header_end - desired_header_y)
        crop_y = min(crop_y, max(0, h - target_h))

        # Nếu footer bị đẩy quá thấp -> kéo crop xuống một chút
        if footer_start - crop_y > int(target_h * 0.80):
            crop_y = min(footer_start - int(target_h * 0.80), max(0, h - target_h))
            crop_y = max(crop_y, 0)

        return scaled.crop((0, crop_y, w, crop_y + target_h))

    def _extend_middle_band(self, scaled: Image.Image, target_h: int) -> Image.Image:
        """Nếu scaled thấp hơn target_h thì kéo giãn phần background giữa (kiểu 'reflow')."""
        w, h = scaled.size
        if h >= target_h:
            return scaled.crop((0, 0, w, target_h))

        # Ưu tiên giữ top text và bottom mascot
        top_h = int(h * 0.32)
        bottom_h = int(h * 0.38)

        # Safety: đảm bảo còn "middle" để kéo giãn
        if top_h + bottom_h >= h - 16:
            top_h = int(h * 0.25)
            bottom_h = int(h * 0.35)

        top_part = scaled.crop((0, 0, w, top_h))
        mid_part = scaled.crop((0, top_h, w, h - bottom_h))
        bottom_part = scaled.crop((0, h - bottom_h, w, h))

        mid_new_h = target_h - top_part.height - bottom_part.height
        if mid_new_h <= 1:
            # Không đủ chỗ -> fallback crop (ít xảy ra)
            return scaled.resize((w, target_h), Image.Resampling.BICUBIC)

        # Stretch background ở giữa
        if mid_part.height < 2:
            # trường hợp ảnh quá "đặc" -> lấy 2px làm slice
            mid_part = scaled.crop((0, top_h, w, min(h, top_h + 2)))

        mid_stretched = mid_part.resize((w, mid_new_h), Image.Resampling.BICUBIC)
        mid_stretched = mid_stretched.filter(ImageFilter.GaussianBlur(radius=0.6))

        canvas = Image.new("RGB", (w, target_h), (255, 255, 255))
        canvas.paste(top_part, (0, 0))
        canvas.paste(mid_stretched, (0, top_part.height))
        canvas.paste(bottom_part, (0, target_h - bottom_part.height))

        # Làm mượt 2 seam (trên và dưới)
        canvas = self._soft_blend_seams(canvas, top_part.height, target_h - bottom_part.height)

        return canvas

    def _soft_blend_seams(self, img: Image.Image, y_seam1: int, y_seam2: int) -> Image.Image:
        """Blend mỏng ở các seam ngang để tránh 'đường kẻ'."""
        try:
            arr = np.array(img).astype(np.float32)
            h, w = arr.shape[:2]
            seam_band = 8

            def blend_band(y0: int):
                y_start = max(0, y0 - seam_band)
                y_end = min(h, y0 + seam_band)
                if y_end <= y_start:
                    return
                band = arr[y_start:y_end].copy()
                blurred = cv2.GaussianBlur(band, (0, 0), sigmaX=1.2)
                # mix 50/50
                arr[y_start:y_end] = (band * 0.6 + blurred * 0.4)

            blend_band(y_seam1)
            blend_band(y_seam2)

            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        except Exception:
            return img

# ==================== MAGIC SWITCH ENGINE ====================
class MagicSwitchEngine:
    """Tự động bố cục lại không cần AI"""
    
    def __init__(self):
        pass
    
    def magic_resize(self, image: Image.Image, elements: List[Dict], 
                    target_size: Tuple[int, int]) -> Tuple[Image.Image, List[Dict]]:
        """
        Magic Switch: Tự động sắp xếp lại các phần tử
        """
        original_w, original_h = image.size
        target_w, target_h = target_size
        
        # Tính toán tỷ lệ scale
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        
        # Chiến lược resize thông minh
        if scale_w > 1.5 or scale_h > 1.5:
            # Mở rộng lớn - dùng Generative Expand
            expand_engine = GenerativeExpandEngine()
            result = expand_engine.expand_image(
                image, target_size,
                context_prompt="professional expansion"
            )
        else:
            # Resize thông minh
            result = self._smart_resize(image, target_size)
        
        # Tính toán vị trí mới cho các phần tử
        new_elements = self._reposition_elements(elements, original_w, original_h, target_w, target_h)
        
        return result, new_elements
    
    def _smart_resize(self, image, target_size):
        """Resize thông minh giữ nội dung quan trọng"""
        w, h = image.size
        target_w, target_h = target_size
        
        # Nếu chỉ thay đổi nhỏ, dùng LANCZOS chất lượng cao
        if abs(w - target_w) < 100 and abs(h - target_h) < 100:
            return image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Content-aware resize đơn giản
        # 1. Xác định vùng quan trọng bằng edge detection
        img_np = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # 2. Tạo importance map từ edges
        importance = edges.astype(np.float32) / 255.0
        
        # 3. Thêm weight cho center
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        center_distance = np.sqrt(((x_coords - center_x) / w)**2 + ((y_coords - center_y) / h)**2)
        center_weight = np.exp(-center_distance * 3)
        importance += center_weight * 0.5
        
        # 4. Resize importance map và ảnh
        importance_resized = cv2.resize(importance, target_size, interpolation=cv2.INTER_LINEAR)
        
        # 5. Simple seam carving simulation (cho prototype)
        # Trong thực tế, bạn có thể implement seam carving thực sự
        # Ở đây dùng resize chất lượng cao với weighting
        
        # Tăng contrast ở vùng quan trọng
        img_resized = cv2.resize(img_np, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Áp dụng importance map để tăng cường vùng quan trọng
        importance_3ch = np.stack([importance_resized] * 3, axis=2)
        img_enhanced = (img_resized * (0.8 + 0.4 * importance_3ch)).clip(0, 255).astype(np.uint8)
        
        return Image.fromarray(img_enhanced)
    
    def _reposition_elements(self, elements, orig_w, orig_h, new_w, new_h):
        """Tính toán vị trí mới cho các phần tử"""
        if not elements:
            return []
        
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        new_elements = []
        for elem in elements:
            elem = elem.copy()
            
            # Scale position
            elem["x"] = int(elem.get("x", 0) * scale_x)
            elem["y"] = int(elem.get("y", 0) * scale_y)
            
            # Scale size với logic thông minh
            if elem.get("type") == "text":
                # Text: scale font size proportionally
                elem["width"] = int(elem.get("width", 100) * scale_x)
                elem["height"] = int(elem.get("height", 50) * scale_y)
                if "font_size" in elem:
                    elem["font_size"] = int(elem["font_size"] * min(scale_x, scale_y))
            
            elif elem.get("type") == "logo":
                # Logo: giữ nguyên tỷ lệ
                scale_min = min(scale_x, scale_y)
                elem["width"] = int(elem.get("width", 100) * scale_min)
                elem["height"] = int(elem.get("height", 100) * scale_min)
            
            else:
                # Generic element
                elem["width"] = int(elem.get("width", 100) * scale_x)
                elem["height"] = int(elem.get("height", 100) * scale_y)
            
            # Đảm bảo không ra ngoài biên
            margin = 20
            elem["x"] = max(margin, min(elem["x"], new_w - elem["width"] - margin))
            elem["y"] = max(margin, min(elem["y"], new_h - elem["height"] - margin))
            
            new_elements.append(elem)
        
        return new_elements

# ==================== CREATIVE AUTOMATION ENGINE ====================
class CreativeAutomationEngine:
    """Creative Automation với template thông minh"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.smart_components = {}
    
    def _load_templates(self):
        """Tải template thông minh"""
        return {
            "modern": {
                "safe_zones": {"top": 0.1, "bottom": 0.1, "left": 0.1, "right": 0.1},
                "grid_columns": 12,
                "grid_rows": 8,
            },
            "minimal": {
                "safe_zones": {"top": 0.15, "bottom": 0.15, "left": 0.15, "right": 0.15},
                "grid_columns": 8,
                "grid_rows": 6,
            },
            "bold": {
                "safe_zones": {"top": 0.05, "bottom": 0.05, "left": 0.05, "right": 0.05},
                "grid_columns": 10,
                "grid_rows": 10,
            }
        }
    
    def create_smart_template(self, image: Image.Image, elements: List[Dict], 
                             template_type: str = "modern"):
        """Tạo template thông minh"""
        template = self.templates.get(template_type, self.templates["modern"])
        
        # Tạo grid system
        grid = self._create_adaptive_grid(image.size, template)
        
        # Gán các phần tử vào grid
        positioned_elements = self._assign_to_grid(elements, grid)
        
        # Tính toán anchor points
        smart_components = self._calculate_anchor_points(positioned_elements, grid)
        
        # Lưu template
        template_id = f"template_{hashlib.md5(image.tobytes()).hexdigest()[:8]}_{int(time.time())}"
        self.smart_components[template_id] = {
            "original_size": image.size,
            "grid": grid,
            "elements": smart_components,
            "template_type": template_type,
        }
        
        return template_id, smart_components
    
    def resize_template(self, template_id: str, target_size: Tuple[int, int]) -> Dict:
        """Resize toàn bộ template thông minh"""
        if template_id not in self.smart_components:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.smart_components[template_id]
        original_size = template["original_size"]
        elements = template["elements"]
        
        # Tính toán scale
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Tạo grid mới
        new_grid = self._create_adaptive_grid(target_size, self.templates[template["template_type"]])
        
        # Áp dụng quy tắc anchor points
        resized_elements = []
        for elem in elements:
            new_elem = self._apply_anchor_rules(elem, scale_x, scale_y, new_grid)
            resized_elements.append(new_elem)
        
        return {
            "size": target_size,
            "elements": resized_elements,
            "requires_expansion": scale_x > 1.5 or scale_y > 1.5,
        }
    
    def _create_adaptive_grid(self, size, template):
        """Tạo grid thích ứng"""
        w, h = size
        cols = template["grid_columns"]
        rows = template["grid_rows"]
        
        col_width = w / cols
        row_height = h / rows
        
        safe_zones = template["safe_zones"]
        safe_area = (
            int(w * safe_zones["left"]),
            int(h * safe_zones["top"]),
            int(w * (1 - safe_zones["right"])),
            int(h * (1 - safe_zones["bottom"]))
        )
        
        return {
            "columns": cols,
            "rows": rows,
            "col_width": col_width,
            "row_height": row_height,
            "safe_area": safe_area,
        }
    
    def _assign_to_grid(self, elements, grid):
        """Gán phần tử vào grid cells"""
        positioned = []
        
        for elem in elements:
            elem_type = elem.get("type", "element")
            
            if elem_type == "text":
                col_span = min(4, grid["columns"] - 2)
                row_span = 1
                start_col = (grid["columns"] - col_span) // 2
                start_row = 1 if "title" in elem.get("role", "") else 3
                
            elif elem_type == "logo":
                col_span = 2
                row_span = 2
                start_col = grid["columns"] - col_span - 1
                start_row = 0
                
            else:
                col_span = 3
                row_span = 3
                start_col = 1
                start_row = 2
            
            positioned.append({
                **elem,
                "grid_position": {
                    "start_col": start_col,
                    "start_row": start_row,
                    "col_span": col_span,
                    "row_span": row_span,
                },
            })
        
        return positioned
    
    def _calculate_anchor_points(self, elements, grid):
        """Tính toán anchor points"""
        anchored = []
        
        for elem in elements:
            grid_pos = elem["grid_position"]
            
            anchors = {
                "horizontal": {
                    "left": grid_pos["start_col"] / grid["columns"],
                    "right": (grid_pos["start_col"] + grid_pos["col_span"]) / grid["columns"],
                },
                "vertical": {
                    "top": grid_pos["start_row"] / grid["rows"],
                    "bottom": (grid_pos["start_row"] + grid_pos["row_span"]) / grid["rows"],
                },
            }
            
            anchored.append({
                **elem,
                "anchors": anchors,
            })
        
        return anchored
    
    def _apply_anchor_rules(self, elem, scale_x, scale_y, new_grid):
        """Áp dụng anchor rules khi resize"""
        anchors = elem["anchors"]
        
        # Tính vị trí mới từ anchor points
        new_cols = new_grid["columns"]
        new_rows = new_grid["rows"]
        
        start_col = max(0, int(anchors["horizontal"]["left"] * new_cols))
        end_col = min(new_cols, int(anchors["horizontal"]["right"] * new_cols))
        start_row = max(0, int(anchors["vertical"]["top"] * new_rows))
        end_row = min(new_rows, int(anchors["vertical"]["bottom"] * new_rows))
        
        col_span = max(1, end_col - start_col)
        row_span = max(1, end_row - start_row)
        
        return {
            **elem,
            "grid_position": {
                "start_col": start_col,
                "start_row": start_row,
                "col_span": col_span,
                "row_span": row_span,
            },
        }

# ==================== MAIN PROCESSING PIPELINE ====================
class BannerResizerUltraLite:
    """Pipeline xử lý chính không cần AI"""
    
    def __init__(self):
        self.generative_engine = GenerativeExpandEngine()
        self.celtra_engine = CeltraReflowEngine()
        self.magic_switch = MagicSwitchEngine()
        self.automation_engine = CreativeAutomationEngine()
    
    def process(self, image: Image.Image, target_sizes: List[Tuple[int, int, str]], 
                mode: str = "auto", elements: List[Dict] = None,
                template_type: str = "modern") -> Dict[str, Any]:
        """
        Xử lý ảnh không cần AI
        """
        results = {}
        
        # Validate input
        if image is None:
            raise ValueError("No image provided")
        
        # Phân tích ảnh
        analysis = ImageAnalyzer.analyze_content(image)
        
        # Tạo template nếu có elements
        template_id = None
        if elements and mode in ["automation", "auto"]:
            try:
                template_id, _ = self.automation_engine.create_smart_template(
                    image, elements, template_type
                )
            except:
                template_id = None
        else:
            elements = elements or []
        
        # Xử lý cho từng kích thước
        for target_w, target_h, size_name in target_sizes:
            print(f"Processing {size_name} ({target_w}x{target_h})...")
            
            target_size = (target_w, target_h)
            
            try:
                # Chọn chiến lược
                src_ratio = image.width / max(1, image.height)
                is_landscape_src = src_ratio >= 1.15
                is_portrait_tgt = (target_h / max(1, target_w)) >= 1.15

                if mode == "celtra" or (mode == "auto" and is_landscape_src and is_portrait_tgt):
                    # Celtra Reflow (No-AI) - phù hợp landscape -> story/vertical
                    result_image = self.celtra_engine.reflow(image, target_size)
                    result_elements = self.magic_switch._reposition_elements(
                        elements, image.width, image.height, target_w, target_h
                    )

                elif mode == "generative" or (mode == "auto" and 
                    (target_w > image.width * 1.5 or target_h > image.height * 1.5)):
                    # Generative Expand (No-AI inpaint)
                    result_image = self.generative_engine.expand_image(
                        image, target_size
                    )
                    result_elements = self.magic_switch._reposition_elements(
                        elements, image.width, image.height, target_w, target_h
                    )

                elif mode == "magic" or (mode == "auto" and elements):
                    # Magic Switch (component-aware)
                    result_image, result_elements = self.magic_switch.magic_resize(
                        image, elements, target_size
                    )

                elif mode == "automation" and template_id:
                    # Creative Automation
                    template_result = self.automation_engine.resize_template(template_id, target_size)

                    if template_result["requires_expansion"]:
                        expanded = self.generative_engine.expand_image(image, target_size)
                        result_image = expanded
                    else:
                        result_image = image.resize(target_size, Image.Resampling.LANCZOS)

                    result_elements = template_result["elements"]

                else:
                    # Smart resize (fallback)
                    result_image = self._smart_resize_fallback(image, target_size)
                    result_elements = elements

# Lưu kết quả
                results[size_name] = {
                    "image": result_image,
                    "elements": result_elements,
                    "size": target_size,
                    "analysis": analysis,
                }
                
            except Exception as e:
                print(f"Warning for {size_name}: {e}")
                # Fallback to simple resize
                try:
                    result_image = image.resize(target_size, Image.Resampling.LANCZOS)
                    results[size_name] = {
                        "image": result_image,
                        "elements": elements,
                        "size": target_size,
                        "analysis": analysis,
                    }
                except:
                    print(f"Failed to process {size_name}")
        
        return results
    
    def _smart_resize_fallback(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize thông minh với padding"""
        w, h = image.size
        target_w, target_h = target_size
        
        # Tính tỷ lệ giữ nguyên
        ratio_w = target_w / w
        ratio_h = target_h / h
        ratio = min(ratio_w, ratio_h)
        
        # Resize với tỷ lệ
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Tạo canvas với màu nền từ ảnh
        canvas = Image.new("RGB", target_size, self._get_background_color(image))
        
        # Đặt ảnh vào giữa
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas.paste(resized, (x_offset, y_offset))
        
        return canvas
    
    def _get_background_color(self, image: Image.Image) -> Tuple[int, int, int]:
        """Lấy màu nền từ ảnh"""
        # Lấy màu từ các cạnh
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        
        # Lấy mẫu từ các cạnh
        samples = []
        
        # Top edge
        if h > 0:
            samples.append(img_np[0, :])
        # Bottom edge
        if h > 1:
            samples.append(img_np[-1, :])
        # Left edge
        if w > 0:
            samples.append(img_np[:, 0])
        # Right edge
        if w > 1:
            samples.append(img_np[:, -1])
        
        if samples:
            all_samples = np.concatenate([s.flatten() for s in samples])
            if len(all_samples) > 0:
                if len(img_np.shape) == 3:
                    # RGB
                    channels = []
                    for i in range(3):
                        channel_samples = all_samples[i::3]
                        if len(channel_samples) > 0:
                            channels.append(int(np.median(channel_samples)))
                        else:
                            channels.append(240)
                    return tuple(channels)
                else:
                    # Grayscale
                    median_val = int(np.median(all_samples))
                    return (median_val, median_val, median_val)
        
        return (240, 240, 240)

# ==================== GRADIO INTERFACE ====================
def create_ultra_lite_interface():
    """Tạo giao diện Gradio siêu nhẹ"""
    
    processor = BannerResizerUltraLite()
    
    with gr.Blocks(title="🚀 Banner Resizer ULTRA LITE", theme=gr.themes.Soft(), css="""
        .gradio-container { max-width: 95% !important; }
        .gr-button { font-weight: bold !important; }
    """) as interface:
        
        # Header
        gr.Markdown("""
        # 🚀 BANNER RESIZER ULTRA LITE
        ### **Professional Banner Generator - No AI Required**
        *Chạy ngay lập tức trên mọi máy tính*
        ---
        """)
        
        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload & Settings")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Design",
                    height=300,
                    interactive=True
                )
                
                with gr.Accordion("🧩 Design Elements (Optional)", open=False):
                    elements_json = gr.Textbox(
                        label="Elements JSON",
                        value='''[
  {"type": "text", "text": "MAIN HEADLINE", "x": 100, "y": 100, "width": 800, "height": 80, "role": "title"},
  {"type": "text", "text": "Subtitle here", "x": 100, "y": 200, "width": 600, "height": 40, "role": "subtitle"},
  {"type": "logo", "x": 900, "y": 50, "width": 150, "height": 150}
]''',
                        lines=6
                    )
                
                mode_selector = gr.Radio(
                    choices=[
                        ("🤖 Auto (Smart Resize)", "auto"),
                        ("📐 Celtra Reflow (No-AI)", "celtra"),
                        ("🎨 Generative Expand", "generative"),
                        ("✨ Magic Switch", "magic"),
                        ("🏢 Template Automation", "automation"),
                        ("🧩 Smart Resize Only", "smart")
                    ],
                    value="auto",
                    label="Processing Mode"
                )
                
                template_selector = gr.Dropdown(
                    choices=["modern", "minimal", "bold"],
                    value="modern",
                    label="Template Style"
                )
                
                gr.Markdown("### 📐 Target Sizes")
                size_presets = gr.CheckboxGroup(
                    choices=[
                        ("Instagram Story (1080x1920)", "1080x1920_story"),
                        ("Instagram Square (1080x1080)", "1080x1080_square"),
                        ("Instagram Portrait (1080x1350)", "1080x1350_portrait"),
                        ("Facebook Cover (1200x630)", "1200x630_fb_cover"),
                        ("Facebook Post (1200x1200)", "1200x1200_fb_post"),
                        ("LinkedIn Post (1200x627)", "1200x627_linkedin"),
                        ("YouTube Thumbnail (1280x720)", "1280x720_yt"),
                        ("Pinterest Pin (1000x1500)", "1000x1500_pinterest"),
                        ("Twitter Header (1500x500)", "1500x500_twitter"),
                        ("Billboard (2000x1000)", "2000x1000_billboard"),
                    ],
                    value=["1080x1920_story", "1080x1080_square", "1200x630_fb_cover"],
                    label="Select Output Sizes"
                )
                
                with gr.Accordion("📏 Custom Size", open=False):
                    with gr.Row():
                        custom_width = gr.Number(value=1200, label="Width", minimum=100)
                        custom_height = gr.Number(value=628, label="Height", minimum=100)
                    custom_name = gr.Textbox(value="Custom Banner", label="Name")
                    add_custom_btn = gr.Button("➕ Add Custom Size", size="sm")
                
                with gr.Accordion("⚙️ Advanced", open=False):
                    quality = gr.Slider(1, 100, value=90, label="Quality")
                    enable_smart_fill = gr.Checkbox(value=True, label="Enable Smart Fill")
                
                generate_btn = gr.Button(
                    "🚀 GENERATE ALL BANNERS",
                    variant="primary",
                    size="lg"
                )
                
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
            
            # Right Panel
            with gr.Column(scale=2):
                gr.Markdown("### 👁️ Results")
                
                gallery = gr.Gallery(
                    label="Generated Banners",
                    columns=3,
                    height=600,
                    object_fit="contain"
                )
                
                with gr.Accordion("📊 Analysis", open=False):
                    analysis_output = gr.JSON(label="Image Analysis")
                
                gr.Markdown("### 💾 Export")
                download_zip = gr.File(label="Download ZIP")
                
                status_text = gr.Markdown("**Status:** Ready")
        
        # State
        custom_sizes_state = gr.State([])
        
        # Add custom size
        def add_custom_size(width, height, name, existing_sizes):
            if not name.strip():
                name = f"Custom_{width}x{height}"
            new_size = (int(width), int(height), name)
            updated = existing_sizes + [new_size]
            return updated, f"✅ Added: {name} ({width}x{height})"
        
        add_custom_btn.click(
            add_custom_size,
            inputs=[custom_width, custom_height, custom_name, custom_sizes_state],
            outputs=[custom_sizes_state, status_text]
        )
        
        # Clear
        def clear_all():
            return None, "", [], "modern", "auto", [], "Cleared", [], {}, None
        
        clear_btn.click(
            clear_all,
            outputs=[
                image_input, elements_json, size_presets, template_selector,
                mode_selector, custom_sizes_state, status_text, gallery,
                analysis_output, download_zip
            ]
        )
        
        # Main processing
        def process_all(image, elements_json_str, mode, template, selected_presets, custom_sizes):
            if image is None:
                return [], {}, None, "❌ Please upload an image"
            
            try:
                # Parse elements
                elements = []
                if elements_json_str and elements_json_str.strip():
                    try:
                        elements = json.loads(elements_json_str)
                    except:
                        pass
                
                # Prepare sizes
                target_sizes = []
                size_map = {
                    "1080x1920_story": (1080, 1920, "Instagram Story"),
                    "1080x1080_square": (1080, 1080, "Instagram Square"),
                    "1080x1350_portrait": (1080, 1350, "Instagram Portrait"),
                    "1200x630_fb_cover": (1200, 630, "Facebook Cover"),
                    "1200x1200_fb_post": (1200, 1200, "Facebook Post"),
                    "1200x627_linkedin": (1200, 627, "LinkedIn Post"),
                    "1280x720_yt": (1280, 720, "YouTube Thumbnail"),
                    "1000x1500_pinterest": (1000, 1500, "Pinterest Pin"),
                    "1500x500_twitter": (1500, 500, "Twitter Header"),
                    "2000x1000_billboard": (2000, 1000, "Billboard"),
                }
                
                for preset in selected_presets:
                    if preset in size_map:
                        target_sizes.append(size_map[preset])
                
                for custom in custom_sizes:
                    target_sizes.append(custom)
                
                if not target_sizes:
                    return [], {}, None, "❌ Please select at least one size"
                
                # Process
                results = processor.process(
                    image, target_sizes, mode, elements, template
                )
                
                # Prepare output
                gallery_items = []
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for size_name, result in results.items():
                        img = result["image"]
                        
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format='PNG', optimize=True)
                        img_data = img_buffer.getvalue()
                        
                        filename = f"{size_name.replace(' ', '_')}.png"
                        zip_file.writestr(filename, img_data)
                        
                        gallery_items.append((img, f"{size_name}"))
                
                # Save zip
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    tmp_file.write(zip_buffer.getvalue())
                    zip_path = tmp_file.name
                
                first_analysis = list(results.values())[0]["analysis"] if results else {}
                
                return gallery_items, first_analysis, zip_path, f"✅ Generated {len(results)} banners"
                
            except Exception as e:
                traceback.print_exc()
                return [], {}, None, f"❌ Error: {str(e)}"
        
        # Connect button
        generate_btn.click(
            process_all,
            inputs=[
                image_input, elements_json, mode_selector, template_selector,
                size_presets, custom_sizes_state
            ],
            outputs=[gallery, analysis_output, download_zip, status_text]
        )
    
    return interface

# ==================== MAIN ====================
def main():
    """Chạy ứng dụng ULTRA LITE"""
    
    print("=" * 70)
    print("🚀 BANNER RESIZER ULTRA LITE")
    print("=" * 70)
    print("📦 Features:")
    print("   • No AI Models Required")
    print("   • Instant Startup")
    print("   • Smart Content-Aware Resize")
    print("   • Generative Expand (Mirror/Inpaint)")
    print("   • Magic Switch Layout")
    print("   • Template Automation")
    print("=" * 70)
    print("💻 Runs on: CPU only - No GPU needed")
    print("=" * 70)
    
    try:
        interface = create_ultra_lite_interface()
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            debug=False,
            show_error=True,
            quiet=True
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()