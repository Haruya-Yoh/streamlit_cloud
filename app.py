import os

# Chromadb の Telemetry を無効化（インポート前に必ずセット）
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "False"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import streamlit as st
from openai import OpenAI

# OpenAI APIキー（Streamlit Cloud の Secrets 設定を利用）
api_key = os.getenv("OPENAI_API_KEY")
client_ai = OpenAI(api_key=api_key)

# ChromaDB クライアントの初期化
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="line_rangers")

# 埋め込みモデルのロード
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_context(question: str, max_len: int = 1800) -> str:
    """
    質問に関連するドキュメントを ChromaDB から取得し、
    最大トークン数を超えないよう連結して返す。
    """
    q_embedding = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=7)

    texts = []
    total_words = 0
    for doc in results["documents"][0]:
        word_count = len(doc.split())
        if total_words + word_count > max_len:
            break
        texts.append(doc)
        total_words += word_count

    return "\n\n###\n\n".join(texts)

def answer_question(question: str, history: list) -> str:
    """
    GPT-4o を使って質問に回答。
    """
    context = create_context(question, max_len=200)
    prompt = f"""
あなたはゲームの攻略情報発信者です。
以下のコンテキストに基づいて、質問に答えてください。
コンテキストに情報がない場合は「わかりません」と答えてください。

コンテキスト:
{context}

---
質問: {question}
回答:""".strip()

    history.append({"role": "user", "content": prompt})
    try:
        resp = client_ai.chat.completions.create(
            model="gpt-4o",
            messages=history,
            temperature=0.7
        )
        answer = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return f"❌ エラー: {e}"

# Streamlit UI セットアップ
st.set_page_config(page_title="攻略AIチャット", page_icon="🎮")
st.title("🎮 攻略AIチャットボット")

# セッション履歴の初期化
if "history" not in st.session_state:
    st.session_state.history = []

# 質問フォーム
question = st.text_input("質問を入力してください：")
if st.button("送信") and question:
    with st.spinner("考え中..."):
        answer = answer_question(question, st.session_state.history)
        st.markdown("---")
        st.markdown(f"**あなたの質問：** {question}")
        st.markdown(f"**AIの回答：** {answer}")
