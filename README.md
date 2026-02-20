# AutoBanner

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
AutoBanner là công cụ **tự bố cục lại banner** để chuyển từ 1 thiết kế gốc sang nhiều kích thước đích (ngang / vuông / dọc), **giữ nguyên tối đa mascot/logo/text** và tạo ra kết quả nhìn như “được thiết kế cho size đích”, không phải chỉ “resize kéo giãn”.

---

## AutoBanner làm được gì?
- Đọc **PSD** và ảnh phẳng (**PNG/JPG/WEBP**).
- Phân loại vai trò phần tử: headline/subheadline/CTA/logo/mascot/hero/background.
- **Phase 2.1 – Adaptive Relayout**: tự tính bố cục theo profile + solver + typography (tự wrap chữ, giữ hierarchy).
- **Phase 3 – Target-first Redesign (Design-native)**: đặt các “điểm neo thương hiệu” (mascot/logo/text/CTA) trước, rồi **tạo lại nền + decor (sky/cityscape/fireworks/confetti)** theo style minh hoạ flat để nhìn như thiết kế cho size đích.
- Có **benchmark** để đo chất lượng, tránh regress.

## AutoBanner KHÔNG phải là gì?
- Không phải Photoshop full-render tất cả layer effects.
- Không phải công cụ “đổi style / đổi concept” theo kiểu thiết kế mới hoàn toàn.
- Không cam kết 100% giống designer cho mọi trường hợp (nhưng Phase 3 đã tối ưu cho banner minh hoạ flat kiểu công ty bạn).

---

## Nên dùng chế độ nào?
- **Phase 3 (khuyến nghị cho banner minh hoạ flat)**: khi bạn đổi ngang → dọc (hoặc tỷ lệ lệch mạnh) và muốn nhìn như thiết kế cho size đó từ đầu.
- **Phase 2.1**: khi bạn cần relayout nhanh, ít thay đổi nền/decor.

---

# Chuẩn bị (cực cơ bản)
Bạn cần:
1) **Git** (để clone repo)
2) **Python 3.11+** (khuyến nghị)
3) (Tuỳ chọn) **Docker Desktop** nếu bạn muốn chạy kiểu Docker

Nếu bạn chưa quen kỹ thuật: chọn **Option A (Docker)** là dễ nhất.

---

# Cài đặt & chạy (dành cho người mới)

## Bước 1 — Clone repo về máy
Mở Terminal/CMD ở chỗ bạn muốn lưu project rồi chạy:

```bash
git clone https://github.com/vansyson1308/smartresizeimage.git
cd smartresizeimage
Nếu lệnh git không chạy: bạn chưa cài Git.

Option A (dễ nhất) — Chạy bằng Docker
Yêu cầu: có Docker Desktop.
docker compose up --build

Sau đó mở trình duyệt:
http://localhost:7860

Dừng app:
nhấn Ctrl + C trong cửa sổ terminal chạy docker.

Option B — Chạy local bằng Python (Windows / macOS / Linux)
Bước 1 — Tạo môi trường ảo (venv)

Windows (CMD):
cd backend
python -m venv .venv
.venv\Scripts\activate

Windows (PowerShell):
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1

macOS / Linux (Terminal):
cd backend
python3 -m venv .venv
source .venv/bin/activate
Khi activate thành công, bạn sẽ thấy trước dòng lệnh có (.venv).

Bước 2 — Cài thư viện
Cài bản dev (đầy đủ lint/test):
pip install -r requirements-dev.txt

Bước 3 — Chạy UI (Gradio)
python -m app.main

Mở trình duyệt:
http://localhost:7860

Cách dùng trong UI (dành cho designer / người không rành code)

Mở UI tại http://localhost:7860

Upload file:

PSD (khuyến nghị)

hoặc PNG/JPG

Chọn chế độ:

Phase 3 (Design-native): khuyến nghị cho banner minh hoạ flat

Phase 2.1: relayout nhanh

Nhập các size cần xuất (ví dụ):

1200×628 (ngang)

1080×1080 (vuông)

1080×1920 (dọc)

Generate → tải kết quả

Nếu bạn dùng ảnh JPG/PNG (flattened)

Phase 3 có thể yêu cầu bạn chọn “anchors” (vùng mascot/text/logo).

UI có preset nhanh cho banner flat: Mascot / MainText / CTA.

CLI smoke test (dành cho người hơi biết kỹ thuật)
Chạy từ thư mục backend/ (đang bật venv).

python - <<'PY'
from app.relayout import ReLayoutEngine

engine = ReLayoutEngine(use_ai=False)
engine.load_file("../path/to/input.png")  # đổi đường dẫn cho đúng file của bạn

for (w, h) in [(1200, 628), (1080, 1080), (1080, 1920)]:
    result = engine.relayout((w, h))
    result.image.save(f"output_{w}x{h}.png")

print("Done. Check output_*.png in current folder.")
PY

Benchmark (đo chất lượng – không commit output)
Chạy từ repo root (khuyến nghị), hoặc từ backend đều được nếu đường dẫn đúng.

python backend/tools/generate_bench_fixtures.py --cases 12 --seed 42
python backend/tools/run_layout_bench.py --mode both --seed 42
python backend/tools/run_layout_bench.py --mode phase3 --seed 42

Output được sinh ra (KHÔNG commit):

backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/before.png

backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/after.png

backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/layout_debug.json

backend/tests/fixtures/outputs/bench_phase21/<case>/<size>/overlay.png

backend/tests/fixtures/outputs/bench_phase21/report.md

Cấu hình (configuration)

File cấu hình chính:

backend/app/config.py

Một số flag đáng chú ý:

LAYOUT_PROFILE_SCORING_ENABLED

LAYOUT_SOLVER_MAX_ITERS

LAYOUT_DEBUG_ENABLED, LAYOUT_DEBUG_DIR

TEXT_SAFE_PLATE_*

Phase 3: các setting liên quan palette/seam/decor/horizon và số candidates

Generative adapter (tuỳ chọn)

Mặc định Phase 3 chạy deterministic (không cần key).

Nếu bạn muốn bật generative adapter (tuỳ setup trong code), bật env var:

Windows (CMD):
set AUTOBANNER_ENABLE_GENERATIVE_REDESIGN=true

macOS/Linux:
export AUTOBANNER_ENABLE_GENERATIVE_REDESIGN=true
Lưu ý: generative adapter là tùy chọn. Không có thì app vẫn chạy được.

Troubleshooting (lỗi hay gặp)
1) OpenCV inpaint failed: cv2 unavailable

Không phải lỗi chết app. App sẽ fallback sang đường deterministic.

Nếu bạn muốn OpenCV hoạt động tốt hơn trong môi trường headless, thử cài requirements-ci.txt.

2) libGL.so.1 (Linux/headless)

Đây là lỗi hệ thống do OpenCV GUI libs.

Cách an toàn: dùng requirements-ci.txt hoặc chạy Docker.

3) Không mở được http://localhost:7860

Kiểm tra terminal có đang chạy python -m app.main không.

Nếu bạn chạy qua proxy hoặc môi trường hạn chế, bật share:

AUTOBANNER_SHARE=true

GRADIO_ANALYTICS_ENABLED=false

4) Benchmark output bị commit nhầm (binary)

Output benchmark không được commit.

Repo đã ignore outputs/ và các cache, nhưng nếu bạn lỡ add, hãy gỡ staged rồi commit lại.

Development (dành cho dev)
# chạy từ repo root (khuyến nghị)
ruff check backend/app backend/tests backend/tools
pytest -q

## License
See [LICENSE](./LICENSE).

## Attribution
Project code is first-party unless noted otherwise in future third-party attribution docs.
