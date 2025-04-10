import re
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
from functools import lru_cache

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

def split_into_questions(text):
    text = text.replace('?', '؟').replace('!', '؟').replace('.', '؟')

    
    question_starters = [
        r'\b(چطور|چگونه|چقدر|چی|چیه|چه|کجا|می‌تونم|میتونم|اگه|اگر|آیا)\b'
    ]

    
    parts = re.split(r'[؟\n]| و ', text)
    parts = [p.strip() for p in parts if p.strip()]

    
    cleaned = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if len(part) < 7 and i + 1 < len(parts):
            combined = f"{part} {parts[i + 1]}"
            cleaned.append(combined.strip())
            i += 2
        else:
            cleaned.append(part)
            i += 1

    
    questions = []
    for part in cleaned:
        splits = re.split(r'(?=' + '|'.join(question_starters) + r')', part)
        for s in splits:
            s = s.strip()
            if len(s) > 6 and any(c.isalpha() for c in s):
                questions.append(s)

    return questions


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

def preprocess_question(user_input):
    user_input = user_input.lower()
    user_input = normalize_text(user_input)
    user_input = re.sub(r'[^\w\s]', '', user_input)
    user_input = re.sub(r'(چه|چی|چطور|چگونه|کجا|چرا|آیا|داره)', '', user_input).strip()
    return user_input

def is_exact_match(user_input, dataset):
    user_input_normalized = preprocess_question(user_input)
    for item in dataset:
        if preprocess_question(item["question"]) == user_input_normalized:
            return item 
    return None

def is_yes_no_question(user_input):
    yes_no_keywords = ["آیا", "مقرون به صرفه", "صرفه"]
    return any(keyword in user_input for keyword in yes_no_keywords)


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

       
        predefined_answers = {
            "فروش متری چیه": "فروش متری یکی از مفاهیم مهم در بازار مسکن، به ویژه در زمینه خرید آپارتمان است. این نوع فروش به خریداران این امکان را می‌دهد که به جای خرید یک واحد کامل، تنها به اندازه نیاز و بودجه خود از یک آپارتمان خرید کنند.",
            "فروش متری خایل": "فروش متری یکی از مفاهیم مهم در بازار مسکن، به ویژه در زمینه خرید آپارتمان است. این نوع فروش به خریداران این امکان را می‌دهد که به جای خرید یک واحد کامل، تنها به اندازه نیاز و بودجه خود از یک آپارتمان خرید کنند."
        }

        
        if user_input in predefined_answers:
            return jsonify({
                "original_input": user_input,
                "answers": [{
                    "input": user_input,
                    "matched_question": user_input,
                    "answer": predefined_answers[user_input],
                    "similarity": 1.0
                }]
            })

        all_questions = split_into_questions(user_input)
        if not all_questions:
            return jsonify({"error": "سوال قابل تشخیص نیست."}), 400

        all_answers = []

        for question in all_questions:
           
            exact_match = is_exact_match(question, dataset)
            if exact_match:
                all_answers.append({
                    "input": question,
                    "matched_question": exact_match["question"],
                    "answer": exact_match["answer"],
                    "similarity": 1.0
                })
                continue 

            # اگر تطابق دقیق نبود، بررسی شباهت‌ها با embeddings
            processed_input = preprocess_question(question)
            user_embedding = get_cached_embedding(processed_input)

            embeddings = [item["embedding"] for item in dataset]
            similarities = np.dot(embeddings, user_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(user_embedding)
            )

            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])
            best_match = dataset[best_index]

            if best_score < 0.5:
                answer_text = "پاسخ مناسبی برای این سوال پیدا نشد."
            else:
                answer_text = best_match["answer"]
                if is_yes_no_question(question):
                    if "مقرون به صرفه" in answer_text or "صرفه" in answer_text:
                        answer_text = "بله، " + answer_text
                    else:
                        answer_text = "خیر، " + answer_text

            all_answers.append({
                "input": question,
                "matched_question": best_match["question"] if best_score >= 0.5 else None,
                "answer": answer_text,
                "similarity": best_score
            })

        return jsonify({
            "original_input": user_input,
            "answers": all_answers
        })

    except Exception as e:
        print("❌ خطا:", e)
        return jsonify({"error": "خطای داخلی سرور", "details": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
