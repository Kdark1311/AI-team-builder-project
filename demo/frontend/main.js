
// --- Phân tích văn bản chính ---
document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const text = document.getElementById("userInput").value.trim();
  const resultEl = document.getElementById("result");
  const explanationEl = document.getElementById("explanation");
  const suggestionEl = document.getElementById("suggestion");

  if (!text) {
    resultEl.innerText = "Bạn chưa nhập văn bản!";
    explanationEl.innerText = "";
    suggestionEl.innerText = "";
    return;
  }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();
    console.log("[LOG] Response:", data);

    resultEl.innerText = data.mbti;
    explanationEl.innerText = data.explanation;
    suggestionEl.innerText = `Vị trí trong team: ${data.role}`;
  } catch (err) {
    console.error(err);
    resultEl.innerText = "Lỗi khi gọi server!";
  }
});
