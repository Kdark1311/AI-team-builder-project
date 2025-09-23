from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import BertTokenizer
import sys
import os
sys.path.append(r"D:\Progamming\Progamming_courses\Quorsk\project\src")
from models import MBTIModel

MODEL_PATH = r"D:\Progamming\Progamming_courses\Quorsk\project\reports\BERT_report\mbti_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Load model
print("Loading model...")
# Load model với config đúng như khi train
# Load model với config giống lúc train
model = MBTIModel(
    model_name="bert-base-uncased",
    dropout=0.4,
    use_hidden_layer=True,   # phải để True
    pooling="cls+mean"       # phải khớp
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded.")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# MBTI labels: 4 chiều (I/E, S/N, T/F, J/P)
MBTI_LABELS = ["I", "E", "S", "N", "T", "F", "J", "P"]

# Dictionary giải thích chi tiết 16 MBTI
mbti_detail = {
    "INTJ": {
        "role": "Người lập kế hoạch",
        "explanation": (
            "INTJ là người tư duy chiến lược, thích lập kế hoạch dài hạn và đặt mục tiêu rõ ràng. "
            "Họ rất độc lập, logic, và thích làm việc với những vấn đề trừu tượng hoặc phức tạp. "
            "Trong team, INTJ thường hoạch định chiến lược, đưa ra quyết định dựa trên lý trí và dữ liệu. "
            "Ưu điểm: tầm nhìn xa, lập kế hoạch tốt. Nhược điểm: đôi khi cứng nhắc, ít chia sẻ cảm xúc."
        )
    },
    "ENTJ": {
        "role": "Người lãnh đạo",
        "explanation": (
            "ENTJ năng động, quyết đoán, lãnh đạo tự nhiên. "
            "Họ thích kiểm soát và điều phối công việc, hướng đến mục tiêu lớn và kết quả cụ thể. "
            "Trong team, ENTJ phân công nhiệm vụ, thúc đẩy tiến độ, giỏi giải quyết xung đột. "
            "Ưu điểm: quyết đoán, chiến lược rõ ràng. Nhược điểm: đôi khi áp đặt, thiếu kiên nhẫn."
        )
    },
    "INFP": {
        "role": "Người giữ giá trị",
        "explanation": (
            "INFP lý tưởng, nhạy cảm, quan tâm giá trị cá nhân và ý nghĩa công việc. "
            "Trong team, INFP giữ vững chuẩn mực, gợi ý ý tưởng sáng tạo, tạo môi trường tích cực. "
            "Ưu điểm: sáng tạo, đồng cảm. Nhược điểm: dễ nhạy cảm, chần chừ khi ra quyết định."
        )
    },
    "ENFP": {
        "role": "Người truyền cảm hứng",
        "explanation": (
            "ENFP nhiệt huyết, sáng tạo, giỏi giao tiếp và truyền cảm hứng. "
            "Trong team, ENFP thúc đẩy tinh thần nhóm, đưa ra ý tưởng đột phá. "
            "Ưu điểm: sáng tạo, nhiệt tình. Nhược điểm: dễ sao nhãng, thiếu tổ chức."
        )
    },
    "ISTJ": {
        "role": "Người tổ chức",
        "explanation": (
            "ISTJ thực tế, tỉ mỉ, tuân thủ quy trình, thích công việc ổn định. "
            "Trong team, họ đảm nhận vai trò thực thi, giám sát tiến độ. "
            "Ưu điểm: trách nhiệm cao, đáng tin cậy. Nhược điểm: khó linh hoạt, đôi khi bảo thủ."
        )
    },
    "ESTJ": {
        "role": "Người điều phối",
        "explanation": (
            "ESTJ quyết đoán, năng động, giỏi tổ chức và quản lý nhóm. "
            "Họ phân công nhiệm vụ, kiểm tra kết quả, đảm bảo team hoàn thành mục tiêu. "
            "Ưu điểm: thực tế, ra quyết định nhanh. Nhược điểm: thiếu linh hoạt, đôi khi áp đặt."
        )
    },
    "ISFJ": {
        "role": "Người hỗ trợ",
        "explanation": (
            "ISFJ tận tâm, chu đáo, quan tâm nhu cầu người khác. "
            "Trong team, họ hỗ trợ chi tiết, tạo môi trường ổn định. "
            "Ưu điểm: trung thành, tỉ mỉ. Nhược điểm: ngại thay đổi, dễ quá tải."
        )
    },
    "ESFJ": {
        "role": "Người kết nối",
        "explanation": (
            "ESFJ thân thiện, hòa đồng, giỏi giao tiếp và tạo kết nối. "
            "Trong team, họ làm cầu nối, hòa giải xung đột. "
            "Ưu điểm: thân thiện, hỗ trợ tốt. Nhược điểm: dễ lo lắng, phụ thuộc ý kiến người khác."
        )
    },
    "INTP": {
        "role": "Người phân tích",
        "explanation": (
            "INTP tò mò, logic, thích phân tích vấn đề trừu tượng. "
            "Trong team, họ giải quyết vấn đề phức tạp, cải thiện quy trình. "
            "Ưu điểm: tư duy phân tích, độc lập. Nhược điểm: khó quyết đoán, ít quan tâm cảm xúc."
        )
    },
    "ENTP": {
        "role": "Người tạo ý tưởng",
        "explanation": (
            "ENTP sáng tạo, năng động, thích thử thách. "
            "Trong team, họ khởi xướng dự án, đưa ra ý tưởng đột phá. "
            "Ưu điểm: sáng tạo, linh hoạt. Nhược điểm: thiếu kiên nhẫn với chi tiết."
        )
    },
    "ISFP": {
        "role": "Người sáng tạo nghệ thuật",
        "explanation": (
            "ISFP nhạy cảm, sáng tạo, thích môi trường tự do. "
            "Trong team, họ mang màu sắc sáng tạo và cảm hứng. "
            "Ưu điểm: sáng tạo, linh hoạt. Nhược điểm: ít chia sẻ cảm xúc, trì hoãn quyết định."
        )
    },
    "ESFP": {
        "role": "Người khuấy động",
        "explanation": (
            "ESFP năng động, hướng ngoại, thích vui vẻ và giao tiếp. "
            "Trong team, họ động viên, tạo sự gắn kết. "
            "Ưu điểm: thân thiện, năng lượng tích cực. Nhược điểm: thiếu lập kế hoạch dài hạn."
        )
    },
    "INFJ": {
        "role": "Người cố vấn",
        "explanation": (
            "INFJ sâu sắc, nhạy cảm, có trực giác tốt. "
            "Trong team, họ cố vấn, hỗ trợ chiến lược và tinh thần nhóm. "
            "Ưu điểm: thấu cảm, tư duy chiến lược. Nhược điểm: quá nhạy cảm, đôi khi cô lập."
        )
    },
    "ENFJ": {
        "role": "Người khuyến khích",
        "explanation": (
            "ENFJ năng động, nhiệt huyết, quan tâm mọi người. "
            "Trong team, họ khích lệ, lãnh đạo tinh thần, truyền cảm hứng. "
            "Ưu điểm: giao tiếp tốt, đồng cảm. Nhược điểm: dễ mệt mỏi khi quan tâm quá nhiều."
        )
    },
    "ISTP": {
        "role": "Người giải quyết sự cố",
        "explanation": (
            "ISTP thực tế, thích khám phá và xử lý vấn đề độc lập. "
            "Trong team, họ giải quyết sự cố, kiểm tra thực tế, tìm giải pháp hiệu quả. "
            "Ưu điểm: nhanh nhạy, độc lập. Nhược điểm: ít chia sẻ cảm xúc, đôi khi thiếu kiên nhẫn."
        )
    },
    "ESTP": {
        "role": "Người hành động",
        "explanation": (
            "ESTP năng động, hướng ngoại, thích hành động ngay. "
            "Trong team, họ khởi xướng hành động, giải quyết vấn đề tức thời. "
            "Ưu điểm: linh hoạt, quyết đoán. Nhược điểm: thiếu kiên nhẫn, đôi khi coi nhẹ kế hoạch dài hạn."
        )
    }
}

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").strip()
    print(f"[LOG] Received text: {text[:50]}...")  # log đầu vào

    if not text:
        return jsonify({
            "mbti": "Unknown",
            "role": "Chưa có",
            "explanation": "Bạn chưa nhập văn bản."
        })

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        pred = (probs > 0.5).astype(int)
        mbti = "".join([MBTI_LABELS[i*2 + p] for i, p in enumerate(pred)])
    

    # ✅ debug in trước khi return
    print("Logits:", logits)
    print("Probs:", probs)
    print("Pred:", pred)
    print(f"[LOG] Predicted MBTI: {mbti}")

    detail = mbti_detail.get(mbti, {"role": "Chưa xác định vai trò", "explanation": "Không có thông tin"})
    role = detail["role"]
    explanation = detail["explanation"]

    print(f"[LOG] Predicted MBTI: {mbti}")
    return jsonify({
        "mbti": mbti,
        "role": role,
        "explanation": explanation
    })
   
if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
