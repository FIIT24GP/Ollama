import csv
import os
import subprocess
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
import numpy as np

# === CONFIG ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "hf.co/t-tech/T-lite-it-1.0-Q8_0-GGUF:Q8_0"
model_path = r"intfloat/multilingual-e5-large"
CSV_LOG = "analitika_history.csv"
RAISS_DB = ['analitika_docx_pdf_html_v2', 'analitika1', 'weld_tks_v1']

# Настройка страницы Streamlit
st.set_page_config(page_title="RAG с FAISS и Ollama", layout="wide")

# Скрываем лишние элементы Streamlit и добавляем кастомный стиль для ответа
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .answer-box {
    background-color: #333333; /* Темный фон */
    color: #ffffff; /* Белый текст */
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #555;
    font-size: 16px;
    line-height: 1.6;
    }
    @keyframes blink {
    50% { opacity: 0; }
    }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


# === Функция для получения списка моделей Ollama ===
@st.cache_data
def models_list():
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    model_names = [line.split()[0] for line in lines[1:] if line.strip()]
    return model_names


ollama_models = models_list()
default_model = MODEL_NAME if MODEL_NAME in ollama_models else ollama_models[0]


# === Загружаем векторную БД ===
@st.cache_resource
def load_vector_db(db_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={'device': 'cpu'})
    return FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)


# === Создаем Ollama LLM ===
def create_ollama_model(model_name, temperature, max_tokens, top_p):
    return OllamaLLM(
        model=model_name,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
    )


# === Функция для ранжирования с BM25 ===
def rerank_with_bm25(query, documents, top_k=5):
    if not documents or not query:
        return [], [0.0] * min(top_k, len(documents))

    tokenized_query = query.split()
    tokenized_docs = [doc.page_content.split() for doc in documents]

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in sorted_indices], [float(scores[i]) for i in sorted_indices]


# === Функция ответа с потоковым выводом ===
def answer_question(prompt, model_name, temperature, max_tokens, top_p, use_rag, db):
    ollama_llm = create_ollama_model(model_name, temperature, max_tokens, top_p)
    russian_prompt = "Отвечай только на русском языке. " + prompt
    answer = ""
    source_text = ""

    if use_rag:
        # Настраиваем память
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        # Получаем документы через ретривер
        retriever = db.as_retriever(search_kwargs={"k": 10})
        sources = retriever.invoke(russian_prompt)

        # Применяем ранжирование с BM25
        reranked_sources, bm25_scores = rerank_with_bm25(russian_prompt, sources, top_k=5)

        # Формируем контекст из ранжированных документов
        context = "\n\n".join([f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(reranked_sources)])
        full_prompt = f"""
        Ты писатель, генерирующий уникальные сценарии, диалоги и квесты для текстовых видеоигр.
        Вопрос: {russian_prompt}
        Контекст:
        {context}
        """

        # Формируем информацию об источниках
        source_info = []
        for i, (doc, score) in enumerate(zip(reranked_sources, bm25_scores), 1):
            filename = doc.metadata.get("source", "Unknown file")
            content = doc.page_content.strip()
            source_info.append(f"[{i}] 📄 **Файл**: {filename} (BM25 Score: {score:.2f})\n\n{content}")
        source_text = "\n\n---\n\n".join(source_info)

        # Потоковый вывод ответа
        for chunk in ollama_llm.stream(full_prompt):
            answer += chunk
            yield chunk
    else:
        # Прямой ответ LLM без RAG
        for chunk in ollama_llm.stream(russian_prompt):
            answer += chunk
            yield chunk
        source_text = "📄 Режим RAG отключён — источники не используются."

    # Сохраняем в CSV
    write_header = not os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Question", "Answer"])
        writer.writerow([prompt, answer])

    # Возвращаем финальный результат
    yield answer, source_text


# === Streamlit UI ===
st.markdown("<h1 style='text-align: center; width: 100%; font-size: 30pt'>📄🔍 RAG технология LLM + FAISS</h1>",
            unsafe_allow_html=True)

st.sidebar.markdown("Задавайте вопросы по базе знаний:")

# Настройки в боковой панели
st.sidebar.header("⚙️ Параметры модели")
model_name = st.sidebar.selectbox("Выбор модели:", ollama_models, index=ollama_models.index(default_model))
selected_db_name = st.sidebar.selectbox("📚 Выберите FAISS базу данных:", RAISS_DB)
temperature = st.sidebar.slider("Температура:", 0.0, 1.0, 0.2, 0.1,
                                help="Чем ближе к 1.0, тем больше креатива у модели")
max_tokens = st.sidebar.slider("Количество токенов:", 200, 8000, 2500, 100,
                               help="Максимальное количество токенов в ответе")
top_p = st.sidebar.slider("Top-P:", 0.0, 1.0, 0.3, 0.1, help="Чем ближе к 1.0, тем больше креатива у модели")
use_rag = st.sidebar.checkbox("🔁 Включить RAG (поиск по базе документов)", value=True)

# Загружаем векторную базу
db = load_vector_db(selected_db_name)

# Ввод вопроса
st.header("💬 Ввод вопроса")
question = st.text_area("Введите вопрос по аналитике:")

if st.button("Поиск"):
    if not question.strip():
        st.warning("Введите вопрос перед отправкой.")
    else:
        with st.spinner("Генерация ответа..."):
            st.subheader("Ответ LLM:")
            # Создаем контейнер для потокового вывода
            with st.container():
                answer_container = st.empty()
                full_answer = ""
                source_text = ""
                # Потоковый вывод ответа
                for item in answer_question(question, model_name, temperature, max_tokens, top_p, use_rag, db):
                    if isinstance(item, str):
                        full_answer += item
                        answer_container.markdown(
                            f'<div class="answer-box">{full_answer}<span style="animation: blink 1s step-end infinite;">|</span></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Финальный результат (answer, source_text)
                        full_answer, source_text = item
                        answer_container.markdown(
                            f'<div class="answer-box">{full_answer}<span style="animation: blink 1s step-end infinite;">|</span></div>',
                            unsafe_allow_html=True
                        )
            # Вывод источников
            st.subheader("📁 Источники:")
            st.text_area("Файлы", value=source_text, height=400)