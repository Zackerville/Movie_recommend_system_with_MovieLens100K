# Bài Tập Lớn — Introduction to AI
## Hệ Gợi Ý Phim (Movie Recommendation System)

## Thông tin môn học

- **Tên môn:** CO3061 Introduction to AI (Nhập môn Trí tuệ Nhân tạo)
- **Học kỳ:** HK251 — Năm học 2025–2026
- **Trường:** Đại học Bách Khoa, ĐHQG-HCM — Bộ môn Khoa học Máy tính
- **GVHD:** TS. Trương Vĩnh Lân
- **File đặc tả:** [mlAssignments_MG.pdf](mlAssignments_MG.pdf) (v1.1, 13/02/2026)

## Thông tin nhóm

| Họ và tên | MSSV | Email | Vai trò |
|-----------|------|-------|---------|
| Ngô Thái Minh Tiến | 2252809 | tien.ngozack2004@hcmut.edu.vn | Trưởng nhóm, phát triển chính, viết báo cáo |

## Mục tiêu

Bài tập lớn này hướng đến việc xây dựng một **hệ gợi ý phim** thông minh, có khả năng nhận vào thông tin của một người dùng cụ thể (lịch sử đánh giá, đặc điểm nhân khẩu học) và đề xuất danh sách Top-K phim phù hợp nhất với sở thích cá nhân của họ. Hệ thống được thiết kế dựa trên dữ liệu thực tế từ tập MovieLens 100K.

Bài tập lớn này hướng đến việc xây dựng một hệ gợi ý phim thông minh, có khả năng nhận vào thông tin của một người dùng cụ thể (lịch sử đánh giá, đặc điểm nhân khẩu học) và đề xuất danh sách Top-K phim phù hợp nhất với sở thích cá nhân của họ. Hệ thống được thiết kế dựa trên dữ liệu thực tế từ tập MovieLens 100K — một trong những bộ dữ liệu chuẩn được sử dụng rộng rãi trong cộng đồng nghiên cứu hệ gợi ý.

Thông qua quá trình thực hiện, đề tài đặt mục tiêu vận dụng và kết hợp một cách tự nhiên các kỹ thuật cốt lõi của trí tuệ nhân tạo, từ tìm kiếm có thông tin để chọn ra danh sách phim tối ưu, qua các thuật toán tối ưu hóa heuristic nhằm đảm bảo sự đa dạng trong gợi ý, đến biểu diễn tri thức bằng hệ luật để mã hóa các quy tắc nghiệp vụ. Đồng thời, đề tài cũng khai thác mô hình xác suất để xử lý yếu tố không chắc chắn trong sở thích người dùng và áp dụng học máy để dự đoán mức độ yêu thích từ dữ liệu lịch sử.

Sản phẩm cuối cùng không chỉ là một mô hình dự đoán đơn lẻ, mà là một hệ thống AI tích hợp hoàn chỉnh, trong đó các thành phần phối hợp với nhau theo một pipeline rõ ràng để biến dữ liệu thô thành những gợi ý có ý nghĩa, có thể giải thích được và thực sự hữu ích cho người dùng cuối.

## Dataset

- **MovieLens 100K** — https://files.grouplens.org/datasets/movielens/ml-100k.zip
- 943 users · 1682 phim · 100,000 ratings (1–5) · metadata: genre, year
- Public, miễn phí, kích thước vừa cho Colab free
- Notebook tự động download và giải nén — không cần mount Drive

## Hướng dẫn chạy

1. Mở Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zackerville/Movie_recommend_system_with_MovieLens100K/blob/main/notebooks/main.ipynb)
2. Chọn `Runtime → Run all`
3. Notebook tự động:
   - Clone repo từ GitHub (lấy modules)
   - Download MovieLens 100K từ nguồn public
   - Chạy toàn bộ pipeline 5 thành phần A–E
   - Hiển thị kết quả và biểu đồ

### Yêu cầu thư viện

Cell đầu notebook tự cài đặt. Các thư viện chính:

```
pandas  numpy  scikit-learn  matplotlib  seaborn  pgmpy
```

## Cấu trúc thư mục

```
btl/
├── notebooks/
│   ├── main.ipynb          # Notebook chính — entry point (front-end)
│   └── 01_eda.ipynb        # EDA dataset MovieLens 100K
├── modules/
│   ├── search/             # A*, BFS, DFS, UCS, heuristic, problem
│   ├── csp/                # Genetic Algorithm, fitness function
│   ├── kb/                 # Rule engine, 20 luật IF-THEN
│   ├── bayes/              # Naive Bayes, Bayes Network
│   ├── ml/                 # Decision Tree, Perceptron, evaluate
│   ├── utils/              # Data loader, feature engineering, split
│   └── pipeline.py         # Pipeline tích hợp 5 thành phần
├── features/               # File đặc trưng (.npy, .pkl)
├── reports/                # Báo cáo PDF + hình ảnh
├── README.md
└── mlAssignments_MG.pdf    # File đặc tả gốc
```

## Liên kết

- 🔗 **Colab Notebook:** [main.ipynb](https://colab.research.google.com/github/Zackerville/Movie_recommend_system_with_MovieLens100K/blob/main/notebooks/main.ipynb)
- 💻 **GitHub:** [Movie_recommend_system_with_MovieLens100K](https://github.com/Zackerville/Movie_recommend_system_with_MovieLens100K)
- 📄 **Báo cáo PDF:** [report.pdf](https://github.com/Zackerville/Movie_recommend_system_with_MovieLens100K/blob/main/reports/report.pdf)

