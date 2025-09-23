import streamlit as st
import sys 
sys.path.append(r"D:\Progamming\Progamming_courses\Quorsk\project\scr_ml")
from models import MBTIClassifierML

# ==============================
# Load model
# ==============================
model = MBTIClassifierML.load(r"D:\Progamming\Progamming_courses\Quorsk\project\reports\mbti_ml.pkl")

# ==============================
# Hàm ánh xạ MBTI -> vai trò trong team
# ==============================
def mbti_to_team_role(mbti_type: str) -> str:
    mbti_type = mbti_type.upper()
    mapping = {
        "ENTJ": "Team Leader - định hướng, quyết đoán",
        "ENFJ": "Coach - truyền cảm hứng, hỗ trợ thành viên",
        "INTJ": "Strategist - người hoạch định chiến lược",
        "INFJ": "Visionary - người định hướng tầm nhìn",
        "ENTP": "Innovator - sáng tạo, đưa ra ý tưởng mới",
        "ENFP": "Motivator - khích lệ, gắn kết đội nhóm",
        "INTP": "Analyst - phân tích dữ liệu, logic",
        "INFP": "Mediator - điều hoà mối quan hệ trong team",
        "ESTJ": "Organizer - quản lý công việc, deadline",
        "ESFJ": "Supporter - chăm lo, kết nối mọi người",
        "ISTJ": "Executor - thực hiện chi tiết, kỷ luật",
        "ISFJ": "Protector - bảo vệ, hỗ trợ hậu cần",
        "ESTP": "Problem-solver - xử lý tình huống nhanh",
        "ESFP": "Entertainer - tạo năng lượng, gắn kết",
        "ISTP": "Tinkerer - giải quyết vấn đề kỹ thuật",
        "ISFP": "Artist - mang lại sự sáng tạo, thẩm mỹ"
    }
    return mapping.get(mbti_type, "Không rõ vai trò")

# ==============================
# Streamlit UI
# ==============================
st.title("🔮 MBTI Personality Predictor")
st.write("Nhập một đoạn văn bản để dự đoán loại MBTI và vai trò trong team.")

user_input = st.text_area("✍️ Viết vài câu mô tả bản thân hoặc cách bạn làm việc trong nhóm:")

if st.button("Dự đoán MBTI"):
    if user_input.strip():
        # Dự đoán từng trục và ghép thành MBTI
        X_tfidf = model.vectorizer.transform([user_input])
        axes = ["IE","NS","TF","JP"]
        mbti = ""
        for axis in axes:
            pred = model.classifiers[axis].predict(X_tfidf)[0]
            if axis == "IE":
                mbti += "I" if pred == 0 else "E"
            elif axis == "NS":
                mbti += "N" if pred == 0 else "S"
            elif axis == "TF":
                mbti += "T" if pred == 0 else "F"
            elif axis == "JP":
                mbti += "J" if pred == 0 else "P"

        role = mbti_to_team_role(mbti)

        st.subheader(f"👉 MBTI dự đoán: **{mbti}**")
        st.success(f"💡 Vị trí phù hợp trong team: {role}")
    else:
        st.warning("⚠️ Vui lòng nhập nội dung để dự đoán.")
