import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

#環境変数の読み込み
load_dotenv()

#APIキーなどの設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "line_rangers"

#OpenAI クライアント初期化
client_ai = OpenAI(api_key=OPENAI_API_KEY)

#Qdrant クライアント初期化
qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

#埋め込みモデル
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#検索
def create_context(question: str, max_len: int = 2100) -> str:
    q_vector = embedder.encode([question])[0]
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vector,
        limit=4
        
    #類似度がこれ以上の時、類似した文書とする。
	score_threshold=0.4
    )

    texts = []
    total_words = 0
    for point in search_result:
        text = point.payload.get("text", "")
        word_count = len(text.split())
        if total_words + word_count > max_len:
            break
        texts.append(text)
        total_words += word_count

    return "\n\n###\n\n".join(texts)

#回答生成
def answer_question(question: str, history: list) -> str:
    context = create_context(question)

    #参照テキストを保存しておく
    st.session_state["last_context"] = context  

    prompt = f"""
あなたはゲームの攻略情報発信者です。
以下のコンテキストに基づいて、質問に答えてください。コンテキストに情報がない場合は「わかりません。YoutubeやXなどのSNSなどに情報がある可能性があるので、調べてみることをおすすめします」と答えてください。

コンテキスト:
{context}

---
質問: {question}
回答:""".strip()

    history.append({"role": "user", "content": prompt})
    try:
        resp = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.7
        )
        answer = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return f"エラー: {e}"

#Streamlit（UI）
st.set_page_config(page_title="LINEレンジャーQ&A")
st.title("LINEレンジャーQ&Aチャットボット")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("質問を入力してください：")

if st.button("送信") and question:
    with st.spinner("考え中..."):
        answer = answer_question(question, st.session_state.history)
        st.markdown("---")
        st.markdown(f"**あなたの質問：** {question}")
        st.markdown(f"**AIの回答：** {answer}")

        #参照したテキストを表示
        if "last_context" in st.session_state:
            st.markdown("### 今回参照したコンテキスト")
            st.text(st.session_state["last_context"])
