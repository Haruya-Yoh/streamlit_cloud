import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# --- Telemetry無効化（ChromaDBが起動時に参照） ---
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "False"

# --- OpenAI初期化（Streamlit Cloudのsecretsから取得） ---
client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- ChromaDB（インメモリ）クライアント初期化 ---
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="line_rangers",
    metadata={"hnsw:space": "cosine"}  # ✅ コサイン類似度で検索
)

# --- 埋め込みモデルをロード（キャッシュ付き） ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- 初期データ登録（CSVファイルから） ---
@st.cache_data
def load_and_register():
    df = pd.read_csv("scraped.csv")

    # タイトル＋本文を連結して文書にする
    if "title" in df.columns and "body" in df.columns:
        texts = [f"{row['title']}：{row['body']}" for _, row in df.iterrows()]
    elif "text" in df.columns:
        texts = df["text"].tolist()
    else:
        raise ValueError("CSVに 'title'+'body' か 'text' カラムが必要です")

    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"id{i}" for i in range(len(texts))]
    )
    return len(texts)

num_docs = load_and_register()

# --- 類似文書の検索と文脈作成 ---
def create_context(question: str, max_len: int = 1800) -> str:
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

# --- GPTで質問に回答 ---
def answer_question(question: str, history: list) -> str:
    context = create_context(question, max_len=200)
    prompt = f"""
あなたはゲームの攻略情報発信者です。
以下のコンテキストに基づいて、質問に答えてください。
コンテキストに情報がない場合は「わかりません」と答えてください。

コンテキスト:
{context}

---
質問: {question}
回答:
""".strip()

    history.append({"role": "user", "content": prompt})
    try:
        response = client_ai.chat.completions.create(
            model="gpt-4o",
            messages=history,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return f"❌ エラー: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="LINEレンジャーQ&A", page_icon="🎮")
st.title("🎮 LINEレンジャーQ&A チャットボット")

st.info(f"📚 登録済み文書数: {num_docs} 件")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("質問を入力してください：")

if st.button("送信") and question:
    with st.spinner("考え中..."):
        answer = answer_question(question, st.session_state.history)
        st.markdown("---")
        st.markdown(f"**あなたの質問：** {question}")
        st.markdown(f"**AIの回答：** {answer}")
