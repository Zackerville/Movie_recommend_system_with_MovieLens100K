# Bài Tập Lớn — Introduction to AI
## Hệ Gợi Ý Phim (Movie Recommendation System)

> **Trạng thái:** Đã chốt đề tài, đang chuẩn bị triển khai. Đọc [CLAUDE.md](CLAUDE.md), [PLAN.md](PLAN.md), [MAPPING.md](MAPPING.md) để biết chi tiết.

## Thông tin môn học

- **Tên môn:** Introduction to AI (Nhập môn Trí tuệ Nhân tạo)
- **Học kỳ:** HK2 — Năm học 2025–2026
- **Trường:** Đại học Bách Khoa, ĐHQG-HCM — Bộ môn Khoa học Máy tính
- **GVHD:** TS. Trương Vĩnh Lân
- **File đặc tả:** [mlAssignments_MG.pdf](mlAssignments_MG.pdf) (v1.1, 13/02/2026)

## Thông tin nhóm

| Họ và tên | MSSV | Email | Vai trò |
|-----------|------|-------|---------|
| Ngô Thái Minh Tiến | 2252809 | tien.ngozack2004@hcmut.edu.vn | Trưởng nhóm, phát triển chính, viết báo cáo |

## Mục tiêu

Bài tập lớn này hướng đến việc xây dựng một **hệ gợi ý phim** thông minh, có khả năng nhận vào thông tin của một người dùng cụ thể (lịch sử đánh giá, đặc điểm nhân khẩu học) và đề xuất danh sách Top-K phim phù hợp nhất với sở thích cá nhân của họ. Hệ thống được thiết kế dựa trên dữ liệu thực tế từ tập MovieLens 100K — một trong những bộ dữ liệu chuẩn được sử dụng rộng rãi trong cộng đồng nghiên cứu hệ gợi ý.

Thông qua quá trình thực hiện, đề tài đặt mục tiêu vận dụng và kết hợp một cách tự nhiên các kỹ thuật cốt lõi của trí tuệ nhân tạo, từ tìm kiếm có thông tin để chọn ra danh sách phim tối ưu, qua các thuật toán tối ưu hóa heuristic nhằm đảm bảo sự đa dạng trong gợi ý, đến biểu diễn tri thức bằng hệ luật để mã hóa các quy tắc nghiệp vụ. Đồng thời, đề tài cũng khai thác mô hình xác suất để xử lý yếu tố không chắc chắn trong sở thích người dùng và áp dụng học máy để dự đoán mức độ yêu thích từ dữ liệu lịch sử.

Sản phẩm cuối cùng không chỉ là một mô hình dự đoán đơn lẻ, mà là một hệ thống AI tích hợp hoàn chỉnh, trong đó các thành phần phối hợp với nhau theo một pipeline rõ ràng để biến dữ liệu thô thành những gợi ý có ý nghĩa, có thể giải thích được và thực sự hữu ích cho người dùng cuối.

## Dataset

- **MovieLens 100K** — https://files.grouplens.org/datasets/movielens/ml-100k.zip
- 943 users · 1682 phim · 100,000 ratings (1–5) · metadata: genre, year
- Public, miễn phí, kích thước vừa cho Colab free
- Cell đầu notebook tự `wget` và unzip — không cần mount Drive

## Cấu trúc thư mục

```
btl/
├── notebooks/              # Colab notebook chính (entry point)
├── modules/                # Python modules
│   ├── search/             # BFS, DFS, UCS, A*
│   ├── csp/                # CSP / GA / hill climbing
│   ├── kb/                 # Knowledge base, logic, IF-THEN
│   ├── bayes/              # Bayes Network / Naive Bayes
│   ├── ml/                 # Decision Tree / Perceptron / NB classifier
│   └── utils/              # Data loading, EDA, plotting
├── features/               # File đặc trưng (.npy / .h5)
├── reports/                # Báo cáo PDF + hình
├── README.md               # File này
├── CLAUDE.md               # Hướng dẫn cho Claude Code
├── PLAN.md                 # Lộ trình chi tiết
├── MAPPING.md              # Mapping chi tiết 5 thành phần A–E vào hệ gợi ý phim
└── mlAssignments_MG.pdf    # File đặc tả gốc
```

## Hướng dẫn chạy

> **TODO** — điền sau khi có notebook:

1. Mở [Colab notebook chính](TODO_LINK_COLAB) trên Google Colab.
2. `Runtime → Run all`. Notebook tự động download dataset từ public source.
3. Xem kết quả trong các cell output.

### Yêu cầu thư viện

```bash
pip install pandas numpy scikit-learn matplotlib seaborn pgmpy experta
```

(Hầu hết đã có sẵn trên Colab; cell đầu notebook sẽ tự cài thêm những gì thiếu.)

## Liên kết

- 📄 Báo cáo PDF: [TODO]
- 🔗 Colab Notebook: [TODO]
- 💻 GitHub: [TODO]

## Tiêu chí chấm

| Hạng mục | Trọng số |
|----------|---------|
| Mô hình hoá bài toán AI | 25% |
| Tìm kiếm / Heuristic / CSP | 20% |
| Bayes / Xác suất | 15% |
| Học máy | 15% |
| Hệ thống tích hợp | 15% |
| Báo cáo & Demo | 10% |
