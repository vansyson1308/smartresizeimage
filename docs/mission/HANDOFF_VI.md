# AutoBanner — Bàn giao (Mission V2, cập nhật 2026-09-11)

Tài liệu này là bản tóm tắt trung thực cho chủ dự án: cái gì đã chạy được và được kiểm chứng ở đâu,
cái gì mới chỉ có code, cái gì bị chặn vì thiếu điều kiện bên ngoài. Sổ cái máy-đọc-được:
`docs/mission/LEDGER.json`. Trạng thái chi tiết: `STATE.md`, `CAPABILITIES.md`, `EXPERIMENTS.md`.

## 1. Vị trí mã nguồn và bằng chứng

- Nhánh: `claude/blissful-archimedes-5n2dvd` trên `vansyson1308/smartresizeimage`; PR nháp #7
  (https://github.com/vansyson1308/smartresizeimage/pull/7). Nhánh chưa được merge; không có
  thao tác nào lên nhánh bảo vệ, không mua dịch vụ, không gửi thiết kế ra ngoài.
- Commit chính của phiên này: `24be1f8` (F1), `18f848f` (F2), `fbe02ae` (F3), `d04b28f` (F4),
  `3b9b329` (F5), `cd4ea58` (docs kiểm toán), `5087bd7` (bảng chiến dịch + chạy tiếp job),
  `3f9d3c0` (hồ sơ thương hiệu), `3fe6858` (gói dịch vụ), `f0e0b7a` (chế độ offline),
  `1f62e64` (grammar + chỉ dẫn sáng tạo), `6ed7ca5` (giữ nguyên chủ thể), `2a6288c` (làm mới
  tăng dần), `bd2c1c7` (tool phản ví dụ), `e27be88` (kết quả ablation grammar).
- Kết quả đo: `docs/mission/results/` (ablation 09-09, H1/H2/H3/H5, `campaign_12x6_2026-09-11`,
  `ablations_grammar_2026-09-11`, `counterexamples_2026-09-11`). Ảnh chụp browser journey và
  `journey_report.json` là artifact của job CI `browser-journey`.

## 2. Chạy và kiểm chứng

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-ci.txt
.venv/bin/pip install "uvicorn[standard]" playwright && .venv/bin/python -m playwright install --with-deps chromium
.venv/bin/ruff check backend                                   # sạch
.venv/bin/python -m pytest backend/tests --ignore=backend/tests/e2e -q   # 343 passed (11/09)
.venv/bin/python -m pytest backend/tests/e2e -rs               # browser journey 42 bước, 0 lỗi, server chạy AUTOBANNER_OFFLINE=1
AUTOBANNER_AUTH=local AUTOBANNER_DATA_DIR=./data .venv/bin/uvicorn backend.app.api.server:app --port 8000
```

Lần chạy đầu server in *setup token*; mở http://localhost:8000 tạo quản trị viên workspace. Docker:
`docker compose up` (chưa build được ở môi trường này vì không có Docker daemon — chưa kiểm chứng).

Đường dẫn chính: `/api/auth/*` (setup, login, members, tokens), `/api/projects` (upload, blank,
elements, document ops, undo), `/api/projects/{id}/variants` (targets, `rows`, `direction`),
`/api/projects/{id}/campaign/rows` (đọc CSV), `/api/projects/{id}/variants/refresh`,
`/api/projects/{id}/export?format=png|jpeg|webp|pdf`, `/api/projects/{id}/export/project`,
`/api/projects/{id}/learned`, `/api/brands/{brand}`, `/api/usage`, `/api/plans`, `/api/health`.
Chi tiết vận hành: `docs/OPERATIONS.md`.

## 3. Năm phát hiện kiểm toán — đã sửa và có test chứng minh

| Phát hiện | Sửa | Test |
|---|---|---|
| F1 archive import tin đường dẫn/metadata | mọi đường dẫn và id do archive kiểm soát được kiểm tra lúc import và lúc dùng; giới hạn số thành viên archive; từ chối symlink; người import thành chủ sở hữu; approval khai trong archive bị reset nếu người import không có quyền duyệt | `test_archive_safety.py` (8) |
| F2 rule cứng bị vi phạm sau khi giữ bố cục cũ nhưng vẫn được chấp nhận | mọi rule cứng được đánh giá trên bố cục cuối (7 loại); vi phạm hoặc không đánh giá được → CRITICAL, chặn chấp nhận; plan tham chiếu vi phạm bị dàn lại | `test_hard_rules.py` (5) |
| F3 text nhiều run chỉ dùng style run đầu | dàn/vẽ theo từng run trên một baseline; PSD lấy đủ style run; giữ run khi sửa, serialize, typography, disclosure export; thuộc tính không hỗ trợ được khai báo | `test_text_runs.py` (12) |
| F4 UI không có hành trình đăng nhập | user cục bộ (PBKDF2), cookie phiên HttpOnly cho cả ảnh/tải về, CSRF header, setup token, vai trò, thành viên, token cá nhân; không bao giờ tắt auth để "sửa" | `test_auth_journey.py` (7), browser journey |
| F5 test không di động | browser journey commit và chạy trên CI; font DejaVu bundled kèm license; test CJK skip theo đo glyph; fixture H3 tự sinh | `tests/e2e`, `test_design_document.py`, `test_local_edits.py` |

## 4. Năng lực: cục bộ / tùy chọn / chưa kiểm chứng

- **Cục bộ, đã kiểm chứng (VERIFIED_LOCAL)**: import PSD (layer tổng hợp) và ảnh phẳng (tách có
  độ tin cậy), tài liệu thiết kế có type, bố cục theo ràng buộc + họ bố cục + grammar, dàn chữ
  font thật với disclosure, kiểm tra chất lượng trên ảnh render (che khuất, cắt, đọc được, rule
  cứng, palette/font thương hiệu, giữ nguyên chủ thể), sửa cục bộ, học từ biến thể đã duyệt và
  lịch sử từ chối, rule thương hiệu, bảng chiến dịch (12 × 6 = 72 đo được), làm mới tăng dần,
  export PNG/JPEG/WebP/PDF + manifest, lưu/mở lại project, auth cục bộ + workspace, gói dịch vụ,
  chạy tiếp job sau restart (đã sửa race giữa bước khôi phục và worker khi cùng ghi index biến
  thể: ghi qua file tạm duy nhất + khóa project), chế độ offline (browser journey chạy dưới chặn mạng).
- **Tùy chọn bên ngoài**: phân loại vai trò bằng CLIP (cần torch, chưa chạy ở đây), OCR bằng
  tesseract (không có → `not_checked`, không bao giờ tính là pass), generative fill (adapter mock,
  không có provider).
- **Chưa kiểm chứng / bị chặn**: build Docker image (không có daemon), PSD thật của khách hàng,
  hiệu chuẩn verdict với người review, đo thời gian sửa của người dùng, corpus 12 thương hiệu
  thật (R1–R4 của phase D), billing/SSO/multi-node.

## 5. Cải thiện đo được so với baseline (fixture tổng hợp, không phải xác nhận khách hàng)

- Hợp đồng chất lượng v2 loại bỏ false positive lịch sử (headline/CTA bị che hoàn toàn nhưng
  benchmark cũ báo PASS); baseline 5/36 chấp nhận, pipeline thiết kế 36/36 (seed 42).
- Bộ lập kế hoạch ràng buộc vs zone: +14 điểm % tuning, +25 điểm % holdout, nửa thời gian.
- H2 lập kế hoạch chung: lỗi nhất quán 9 → 3 trên 60 lần, cùng tỷ lệ chấp nhận.
- H1 học từ 1 ví dụ đã duyệt/hướng: đồng thuận với bố cục của designer trên size giữ lại 0,24 → 0,85;
  size phải dàn lại 60/60 → 6/60.
- H3 sửa cục bộ: 180/180 lần sửa không đổi pixel ngoài phần tử được sửa.
- H5 rule từ từ chối: lặp lại từ chối 28 → 0 trên 14 chiến dịch sau (reviewer theo luật).
- Chiến dịch 12 dòng × 6 định dạng: 72/72 render trong 88 s (2 worker), 65 chấp nhận, 5 cần
  review (OCR nghi ngờ một headline tiếng Việt), 2 fail trung thực (subheadline dài không vừa
  300×250 ở sàn 16 px).
- H6 grammar bố cục: **không** tăng chấp nhận trên corpus này (0,89 / 0,92 cả hai nhánh), họ grammar
  thắng 6/60; giữ bật để đáp ứng chỉ dẫn sáng tạo, không tuyên bố là cải thiện chất lượng.
- Dải cực hẹp 728×90: đo trên corpus phát hiện check giữ chủ thể báo sai logo 22×10 px (làm tròn số nguyên) — đã sửa, chấp nhận 3/15 → 12/15; ba trường hợp copy dài còn lại được xử lý bằng quyền "có thể bỏ ở kích thước nhỏ" do người thiết kế bật cho subheadline (planner ghi quyết định `dropped:…`, check nhất quán họ đánh dấu để review) → cả ba thành `accepted`.
- Tìm phản ví dụ bằng oracle độc lập (420 render, có OCR): 0 render được chấp nhận mà oracle bác bỏ, 0 fail không giải thích được; 20 lần oracle bật đều trùng verdict không chấp nhận. Phát hiện kèm theo: bước repair có thể làm hỏng rule cứng về thứ tự mà planner đã tuân thủ (đã sửa, `acfeebd`); copy dài hoặc rule cứng ở kích thước nhỏ đẩy chồng chữ vượt canvas — planner nay phạt theo phần diện tích vượt canvas (`9b6e0c8`): cùng 180 render, chấp nhận 145 → 153, oracle bounds bật 15 → 10; phần còn lại là copy không vừa bất kỳ họ nào và fail đúng lý do.

## 6. Mức hoàn thiện thương mại và điểm chặn

Đã có: quy trình khai báo chạy trọn (brand → 12 dòng → 72 biến thể → review theo dòng → sửa
có phạm vi → export → lưu/restart/mở lại), offline sau cài đặt, workspace thứ hai cách ly, người
duyệt bằng tài khoản khác; vận hành: backup/restore, retention, rate limit, quota/plan, log sự kiện.
Chưa có (đúng như đã ghi): billing thật, SSO, chạy nhiều node, build Docker được kiểm chứng,
kiểm chứng trên thiết kế thật và người review thật. Không thao tác nào cần tiền hay quyền
ngoài phạm vi được thực hiện; các bước cần phê duyệt cuối (merge, launch) để lại cho chủ dự án.
