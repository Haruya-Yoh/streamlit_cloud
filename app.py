import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import streamlit as st
from openai import OpenAI

# OpenAI APIキー（Streamlit CloudのSecrets設定を利用）
api_key = os.getenv("OPENAI_API_KEY")
client_ai = OpenAI(api_key=api_key)

# Chromaクライアントの初期化
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="line_rangers")

# 埋め込みモデルのロード
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def create_context(question, max_len=1800):
    """
    ユーザーの質問に対して、関連する文書をChromaから取得し、コンテキストを構築する。
    """
    q_embedding = embedder.encode([question])[0]

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=7
    )

    returns = []
    cur_len = 0

    for doc in results["documents"][0]:
        cur_len += len(doc.split())
        if cur_len > max_len:
            break
        returns.append(doc)

    return "\n\n###\n\n".join(returns)

def answer_question(question, conversation_history):
    """
    GPT APIを使って質問に答える関数。
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

    conversation_history.append({"role": "user", "content": prompt})

    try:
        response = client_ai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return f"❌ エラー: {e}"

# Streamlit UI
st.set_page_config(page_title="攻略AIチャット", page_icon="🎮")
st.title("🎮 攻略AIチャットボット")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("質問を入力してください：")

if st.button("送信") and question:
    with st.spinner("考え中..."):
        answer = answer_question(question, st.session_state.history)
        st.markdown("---")
        st.markdown(f"**あなたの質問：** {question}")
        st.markdown(f"**AIの回答：** {answer}")
