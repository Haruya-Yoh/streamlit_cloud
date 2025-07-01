import os
import pandas as pd
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 1) APIキー取得
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client_ai = OpenAI(api_key=api_key)

# 2) 埋め込みモデルをキャッシュ
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# 3) インメモリChromaDBとデータ登録をキャッシュ
@st.cache_resource
def load_collection_and_data():
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="line_rangers", metadata={"hnsw:space": "cosine"}
    )

    # 既に登録済みなら追加登録をスキップ
    try:
        if collection.count() > 0:
            return collection
    except Exception:
        # count() が未サポートの場合はスキップせず処理を続行
        pass

    df = pd.read_csv("scraped.csv")
    if "title" in df.columns and "body" in df.columns:
        texts = [f"{r['title']}：{r['body']}" for _, r in df.iterrows()]
    elif "text" in df.columns:
        texts = df["text"].tolist()
    else:
        raise ValueError("CSV に 'title'+'body' または 'text' カラムが必要です。")

    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"id{i}" for i in range(len(texts))]
    )
    return collection

collection = load_collection_and_data()

# 4) コンテキスト構築・回答生成
def create_context(question: str, max_len: int = 1800) -> str:
    q_emb = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=7)
    docs, total = [], 0
    for doc in results["documents"][0]:
        wc = len(doc.split())
        if total + wc > max_len:
            break
        docs.append(doc)
        total += wc
    return "\n\n###\n\n".join(docs)

def answer_question(question: str, history: list) -> str:
    context = create_context(question, max_len=200)
    prompt = f"""
あなたはゲームの攻略情報発信者です。
以下のコンテキストに基づいて、質問に答えてください。
情報がなければ「わかりません」とだけ答えてください。

コンテキスト:
{context}

---
質問: {question}
回答:
""".strip()

    history.append({"role": "user", "content": prompt})
    try:
        res = client_ai.chat.completions.create(
            model="gpt-4o", messages=history, temperature=0.7
        )
        ans = res.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": ans})
        return ans
    except Exception as e:
        return f"❌ エラー: {e}"

# 5) Streamlit UI
st.set_page_config(page_title="LINEレンジャーQ&A", page_icon="🎮")
st.title("🎮 LINEレンジャーQ&A チャットボット")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("質問を入力してください：")
if st.button("送信") and q:
    with st.spinner("回答を生成中..."):
        a = answer_question(q, st.session_state.history)
        st.markdown("---")
        st.markdown(f"**あなたの質問：** {q}")
        st.markdown(f"**AIの回答：** {a}")
