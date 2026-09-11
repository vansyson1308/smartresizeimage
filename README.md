# AutoBanner

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
AutoBanner là công cụ **tự bố cục lại banner** để chuyển từ 1 thiết kế gốc sang nhiều kích thước đích (ngang / vuông / dọc), **giữ nguyên tối đa mascot/logo/text** và tạo ra kết quả nhìn như “được thiết kế cho size đích”, không phải chỉ “resize kéo giãn”.

---

## AutoBanner làm được gì?
- Đọc **PSD** (giữ text native, layer, opacity, blend mode cơ bản) và ảnh phẳng (**PNG/JPG/WEBP** – nhập như một bức ảnh duy nhất, luôn cần review).
- Hoặc **dựng master từ asset rời**: canvas trống + thêm ảnh (logo, sản phẩm) + thêm text.
- Phân loại vai trò phần tử (headline/CTA/logo/hero/background…) kèm **độ tin cậy** để bạn xác nhận hoặc sửa.
- Đề xuất **quy tắc** (rule/constraint) có thể duyệt: phải luôn hiển thị, khoảng trống quanh logo, giữ nhóm, cho phép chồng lấn có chủ ý, cỡ chữ tối thiểu.
- **Học từ biến thể đã duyệt**: sau khi bạn duyệt một biến thể, các lần sinh tiếp theo cùng hướng (ngang/vuông/dọc) đi theo bố cục đó; quy tắc suy ra kèm độ tin cậy được hiển thị để kiểm tra (không huấn luyện mô hình, không gửi thiết kế ra ngoài).
- **Sửa cục bộ khi đổi chiến dịch**: đổi copy/asset/màu trên một biến thể đã sinh chỉ thay đổi đúng phần tử đó, bố cục còn lại giữ nguyên (tùy chọn "Keep layout", bật mặc định); nếu copy mới không vừa ô cũ, hệ thống dàn lại và báo rõ `layout_change`.
- **Rule từ lịch sử từ chối**: khi bạn từ chối một biến thể kèm lý do ("logo quá nhỏ", "CTA khó đọc") rồi duyệt bản sửa, hệ thống đề xuất rule tương ứng kèm bằng chứng để bạn thêm bằng một cú nhấp; lý do không hiểu được thì chỉ ghi nhận, không bịa rule.
- **Hồ sơ thương hiệu**: lưu bảng màu, font tiêu đề/thân, khoảng trống và cỡ tối thiểu của logo, tone giọng; project cùng brand nhận các rule này dưới dạng đề xuất, mọi biến thể được kiểm tra màu chữ và font so với hồ sơ (lệch brand → `needs_review`, không bao giờ im lặng).
- **Gói dịch vụ (entitlement) và chạy offline**: mỗi workspace thuộc một gói (free/team/business/unlimited hoặc gói tự định nghĩa) giới hạn biến thể/ngày, project, thành viên, số dòng chiến dịch; `GET /api/usage` cho biết đã dùng/còn lại, vượt hạn mức trả về 402 kèm tên gói. Đặt `AUTOBANNER_OFFLINE=1` để chặn mọi kết nối ra ngoài ở tầng socket: toàn bộ quy trình (kể cả browser journey trong CI) chạy không cần mạng.
- **Rule theo thương hiệu**: rule đã xác nhận (tự thêm hoặc từ lịch sử từ chối) trong một project được đề xuất lại cho các project cùng brand của bạn, có nhãn nguồn để duyệt hoặc bỏ.
- Sinh biến thể theo nhiều kích thước với **text được dàn lại bằng font thật** (không kéo giãn raster), rồi **kiểm tra trên ảnh render** (phần tử có bị che/cắt không, chữ có đọc được không) và tự sửa trong phạm vi giới hạn.
- **Bảng chiến dịch**: dán bảng CSV/TSV (mỗi dòng một thông điệp: headline, CTA, ngôn ngữ…) hoặc thêm dòng bằng tay; mỗi dòng được sinh ở mọi kích thước đã chọn trong một job (ví dụ 12 dòng × 6 định dạng = 72 biến thể), review lọc theo dòng, export có một thư mục cho mỗi dòng; copy verbatim (giá, pháp lý) không bao giờ bị bảng ghi đè.
- Review theo verdict `accepted / needs_review / failed`, duyệt/từ chối kèm lý do, sửa copy cho riêng một biến thể, xuất PNG/JPEG/WebP/PDF (một trang mỗi biến thể) kèm manifest, và **lưu project để mở lại**. Job đang chạy dở khi server khởi động lại sẽ được **chạy tiếp** từ brief đã lưu.
- Có benchmark chạy đúng đường production để đo chất lượng, không tự lừa mình.

## AutoBanner KHÔNG phải là gì?
- Không phải Photoshop full-render tất cả layer effects (effect không hỗ trợ được ghi rõ trong import notes).
- Chưa tách ảnh phẳng thành layer; ảnh phẳng chỉ được co giãn thông minh và luôn ở trạng thái cần review.
- Có đăng nhập cục bộ nhiều người dùng và tách workspace, nhưng chưa có SSO, billing, hay tích hợp nền tảng quảng cáo. Xem `docs/mission/CAPABILITIES.md` để biết trạng thái thật của từng tính năng.

---

## Chạy nhanh (API + giao diện web mới)

```bash
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # hoặc requirements-ci.txt để chạy test
sudo apt-get install tesseract-ocr       # tuỳ chọn: bật kiểm tra OCR (nếu thiếu, check ghi not_checked)
cd .. && AUTOBANNER_DATA_DIR=./data uvicorn backend.app.api.server:app --port 8000
```

Mở http://localhost:8000 → kéo thả PSD/PNG (hoặc tạo canvas trống) → kiểm tra phần tử & quy tắc → chọn kích thước → Generate → Review → Export. Tài liệu API: http://localhost:8000/api/docs.

Đăng nhập và phân quyền (chế độ triển khai): đặt `AUTOBANNER_AUTH=local`; lần chạy đầu server in ra một *setup token*, mở giao diện web và tạo quản trị viên của workspace (không gọi dịch vụ ngoài). Quản trị viên thêm thành viên với vai trò `viewer` / `editor` / `approver` / `admin` ngay trong thẻ Workspace; mỗi người dùng có thể tạo API token cá nhân cho script/CI (header `X-API-Key`). Mỗi workspace chỉ thấy project/job của mình. Không đặt gì cả = chế độ *open* (chỉ để phát triển; giao diện hiện cảnh báo). Hạn mức: `AUTOBANNER_QUOTA_VARIANTS_PER_DAY`, `AUTOBANNER_QUOTA_PROJECTS`. Vận hành, sao lưu, giới hạn: xem `docs/OPERATIONS.md`.

Ảnh phẳng (PNG/JPG) được **tách thử** thành nền + khối chữ (OCR) + chủ thể; mọi phần tử tách ra đều đánh dấu `recovered` kèm độ tin cậy, giữ dạng raster cho đến khi bạn bấm "Convert to editable text". Biến thể sinh từ phần tử chưa xác nhận luôn ở trạng thái `needs_review`.

Khi sinh nhiều kích thước trong một lần, hệ thống chọn **một bố cục cho mỗi hướng** (ngang/vuông/dọc) và kiểm tra tính nhất quán giữa các biến thể (cùng phần tử, cùng thứ tự đọc, cùng thang phân cấp chữ).

Giao diện Gradio cũ vẫn chạy được bằng `python -m backend.app.main` (cổng 7860), dùng cho relayout nhanh Phase 2.1 / Phase 3.

---

## Nên dùng chế độ nào?
- **Giao diện web mới (khuyến nghị)**: quy trình đầy đủ, text native, có kiểm tra chất lượng và lưu project.
- **Phase 3 (Gradio)**: banner minh hoạ flat, đổi tỷ lệ mạnh, muốn tái tạo nền theo kiểu flat illustration (bộ sinh nền hiện là thủ tục/deterministic, chưa có model sinh ảnh thật).
- **Phase 2.1 (Gradio)**: relayout nhanh, ít thay đổi nền.

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

Benchmark chạy đúng đường production (`ReLayoutEngine`) và chấm bằng **quality contract v2**
(`backend/app/quality`): kiểm tra trên ảnh render thật (phần tử có bị che/cắt không, chữ có đọc
được bằng OCR không, thiếu phần tử bắt buộc, sai kích thước xuất). Mỗi biến thể nhận verdict
`accepted` / `needs_review` / `failed`; check không chạy được ghi `not_checked` và không bao giờ
được tính là đạt. Cấu hình (seed, số candidate Phase 3, môi trường, commit) được ghi trong
`summary.json`. Muốn có OCR: cài `tesseract-ocr` (apt) và `pip install pytesseract`.

Trạng thái thực tế của từng tính năng và bằng chứng: xem `docs/mission/CAPABILITIES.md`.

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
