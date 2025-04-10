import re
import numpy as np
from flask import Flask, json, request, jsonify
from sentence_transformers import SentenceTransformer
import os
from functools import lru_cache
import requests
import json

app = Flask(__name__)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@lru_cache(maxsize=1000)
def get_cached_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def normalize_text(text):
    slang_dict = {
        "میصرفه": "مقرون به صرفه است",
        "می‌صرفه": "مقرون به صرفه است",
        "چیه": "چیست",
        "چطوری": "چگونه",
        "روشش چیه": "نحوه آن چیست",
        "فایده داره": "مزایا دارد",
        "بدیش چیه": "معایب آن چیست"
    }
    for slang, formal in slang_dict.items():
        text = text.replace(slang, formal)
    return text

def preprocess_question(user_input):
    user_input = user_input.lower()
    user_input = normalize_text(user_input)
    user_input = re.sub(r'[^\w\s]', '', user_input)
    user_input = re.sub(r'(چه|چی|چطور|چگونه|کجا|چرا|آیا|داره)', '', user_input).strip()
    return user_input

def load_dataset(file_path):
    qa_pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        qa_blocks = re.split(r'\n{2,}', content)
        for block in qa_blocks:
            lines = block.split("\n")
            if len(lines) >= 2:
                question = lines[0].strip()
                answer = " ".join(lines[1:]).strip()
                embedding = model.encode(question, convert_to_numpy=True)
                qa_pairs.append({"question": question, "answer": answer, "embedding": embedding})
    return qa_pairs

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.txt")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npy")

dataset = load_dataset(DATASET_PATH)
np.save(EMBEDDINGS_PATH, [item["embedding"] for item in dataset])
print("✅ Embeddings ساخته شد.")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_input = request.json.get("question", "").strip()
        if not user_input:
            return jsonify({"error": "لطفاً سوالی وارد کنید."}), 400

        user_input_normalized = preprocess_question(user_input)
        best_match = None
        max_similarity = -1

        for item in dataset:
            similarity = np.dot(item["embedding"], get_cached_embedding(user_input_normalized))
            if similarity > max_similarity:
                best_match = item
                max_similarity = similarity

        # پاسخ اولیه (از دیتاست یا خالی)
        raw_answer = best_match["answer"] if best_match and max_similarity > 0.5 else ""

        # ساخت prompt اختصاصی برای API خارجی
        final_prompt = (
            f"متن:\n{raw_answer}\n\n"
            f"سؤال: {user_input}\n\n"
            f"لطفاً به زبان فارسی پاسخ بده:\n"
            f"پاسخ را فقط به زبان فارسی بده و از هیچ کلمه انگلیسی استفاده نکن.\n"
            f"اگر سوال به متن بالا مرتبط نبود، بگو 'اطلاعاتی ندارم'.\n"
            f"اگر از کلاهبرداری بودن پرسید، بگو 'خیر، شرکت هومینگر به شما این تضمین را می‌دهد که این کار کلاهبرداری نیست'.\n"
            f"اگر پرسید این کار سودآور است یا نه، بگو 'بله، این کار سودآور است'.\n"
            f"لطفاً جواب را به صورت مختصر و واضح بده."
        )

        # ارسال به API خارجی
        api_url = "http://46.167.136.122:11434/api/generate"
        data = {
            "model": "partai/dorna-llama3",
            "prompt": final_prompt
        }

        response = requests.post(api_url, json=data, stream=True)
        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = line.decode('utf-8')
                    json_response = json.loads(json_response)
                    response_text += json_response.get('response', '')
                    if json_response.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        return jsonify({
            "original_input": user_input,
            "answers": [{
                "input": user_input,
                "matched_question": best_match["question"] if best_match else "پاسخی یافت نشد",
                "answer": response_text or 'پاسخ موجود نیست',
                "similarity": float(max_similarity)
            }]
        })

    except Exception as e:
        print("❌ خطا:", e)
        return jsonify({"error": "خطای داخلی سرور", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
