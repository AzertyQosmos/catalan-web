import streamlit as st
import json
import re
from datetime import datetime, timedelta
from groq import Groq

# ====== НАСТРОЙКИ ======
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-safeguard-20b",
]
MODEL_DISPLAY_NAMES = {
    "openai/gpt-oss-120b": "GPT-oss-120b (мощная)",
    "openai/gpt-oss-20b": "GPT-oss-20b (быстрая)",
    "groq/compound": "Groq Compound",
    "groq/compound-mini": "Groq Compound Mini",
    "openai/gpt-oss-safeguard-20b": "GPT-OSS Safeguard",
}
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_DESCRIPTIONS = {
    "A1": "beginner. Use ONLY the most basic vocabulary. VERY short sentences (3-6 words). Present tense ONLY.",
    "A2": "elementary. Simple everyday topics. Mostly present and simple past. Short connected sentences.",
    "B1": "intermediate. Opinions, experiences, travel. Past tenses. Some subjunctive.",
    "B2": "upper intermediate. Abstract topics. Wide range of tenses including subjunctive.",
    "C1": "advanced. Nuanced, literary, academic topics. Sophisticated vocabulary.",
    "C2": "mastery. Highly nuanced, poetic or technical. Mastery of register.",
}
SRS_INTERVALS = [1, 3, 7, 14, 30, 60, 120]

# ====== ИНИЦИАЛИЗАЦИЯ SESSION STATE ======
if "vocab" not in st.session_state:
    st.session_state.vocab = []
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_parsed_words" not in st.session_state:
    st.session_state.last_parsed_words = []
if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0
if "flashcard_show_answer" not in st.session_state:
    st.session_state.flashcard_show_answer = False
if "session_correct" not in st.session_state:
    st.session_state.session_correct = 0
if "session_incorrect" not in st.session_state:
    st.session_state.session_incorrect = 0
if "in_session" not in st.session_state:
    st.session_state.in_session = False


# ====== МЕНЕДЖЕР СЛОВАРЯ (работает через session_state) ======
def save_vocab():
    """Словарь хранится в session_state. При желании можно экспортировать в JSON."""
    pass


def add_word(word, translation, level="A1"):
    word_clean = word.strip().lower()
    if not word_clean:
        return False
    for w in st.session_state.vocab:
        if w["word"].lower() == word_clean:
            w["translation"] = translation.strip()
            w["level"] = level
            return False
    st.session_state.vocab.append({
        "word": word.strip(),
        "translation": translation.strip(),
        "level": level,
        "added": datetime.now().isoformat(timespec="seconds"),
        "next_review": datetime.now().strftime("%Y-%m-%d"),
        "interval_index": 0,
        "correct": 0,
        "incorrect": 0,
    })
    return True


def add_words_batch(pairs, level="A1"):
    added = 0
    for w, t in pairs:
        if add_word(w, t, level):
            added += 1
    return added


def remove_word(word):
    st.session_state.vocab = [w for w in st.session_state.vocab if w["word"] != word]


def get_due_words():
    today = datetime.now().date()
    due = []
    for w in st.session_state.vocab:
        try:
            review_date = datetime.fromisoformat(w["next_review"]).date()
            if review_date <= today:
                due.append(w)
        except Exception:
            due.append(w)
    return due


def mark_correct(word):
    for w in st.session_state.vocab:
        if w["word"] == word:
            w["correct"] = w.get("correct", 0) + 1
            idx = min(w.get("interval_index", 0), len(SRS_INTERVALS) - 1)
            days = SRS_INTERVALS[idx]
            w["next_review"] = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            if idx < len(SRS_INTERVALS) - 1:
                w["interval_index"] = idx + 1
            break


def mark_incorrect(word):
    for w in st.session_state.vocab:
        if w["word"] == word:
            w["incorrect"] = w.get("incorrect", 0) + 1
            w["interval_index"] = 0
            w["next_review"] = datetime.now().strftime("%Y-%m-%d")
            break


def count_due():
    return len(get_due_words())


# ====== ПАРСИНГ СЛОВАРЯ ИЗ ТЕКСТА ======
def parse_vocabulary(text):
    if not text:
        return []
    match = re.search(r"###\s*VOCABULARY\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    vocab_block = match.group(1)
    pairs = []
    pattern = re.compile(r"^\s*[-•\*]?\s*(.+?)\s*[—–\-:]\s*(.+?)\s*$", re.MULTILINE)
    for m in pattern.finditer(vocab_block):
        word = m.group(1).strip().strip("•-*")
        translation = m.group(2).strip()
        if word and translation and len(word) < 80:
            pairs.append((word, translation))
    return pairs


# ====== GROQ API ======
def generate_text(api_key, model, topic, level, include_vocab, include_translation):
    client = Groq(api_key=api_key)
    if level == "A1":
        prompt = f"""You are an expert Catalan language teacher. Generate a VERY SIMPLE text in CATALAN about: "{topic}".
CEFR level: A1. Sentences 12 words or less. Simple times ONLY. No complex grammar. 10-15 sentences max.
Output format:
### TEXT
[Catalan text]
### VOCABULARY
[list of A1 words: Catalan — Russian translation]
"""
        if include_translation:
            prompt += "\n### TRANSLATION\n[Russian translation]"
    else:
        prompt = f"""You are an expert Catalan language teacher. Generate a text in CATALAN about: "{topic}".
CEFR level: {level} — {CEFR_DESCRIPTIONS[level]}. Length: 200-300 words. Natural and grammatically correct.
Output format:
### TEXT
[Catalan text]
### VOCABULARY
[list of {level} words: Catalan — Russian translation]
"""
        if not include_vocab:
            prompt += "\n(omit VOCABULARY section)"
        if include_translation:
            prompt += "\n### TRANSLATION\n[full Russian translation]"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=3000,
    )
    return response.choices[0].message.content


def translate_word(api_key, model, word, context_text=""):
    client = Groq(api_key=api_key)
    context_section = ""
    if context_text and context_text.strip():
        ctx = context_text.strip()[:800]
        context_section = f"\nThe word appears in this Catalan text:\n---\n{ctx}\n---\n"
    prompt = f"""You are an expert Catalan-Russian dictionary.
Translate the Catalan word/phrase: "{word}"
{context_section}
Use EXACTLY this format:
### ПЕРЕВОД
[word] — [Russian translation]
### ГРАММАТИКА
[part of speech, gender, etc. Brief]
### В КОНТЕКСТЕ
[how this word is understood in the text above. If no context — write "Контекст не предоставлен"]
### ПРИМЕРЫ
1. [Catalan] — [Russian]
2. [Catalan] — [Russian]
3. [Catalan] — [Russian]
All explanations in Russian."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content


def extract_translation_pair(text):
    match = re.search(r"###\s*ПЕРЕВОД\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    block = match.group(1).strip()
    m = re.search(r"(.+?)\s*[—–\-:]\s*(.+)", block)
    if m:
        word = m.group(1).strip().strip("[]")
        translation = m.group(2).strip().strip("[]")
        if word and translation:
            return (word, translation)
    return None


# ====== НАСТРОЙКА СТРАНИЦЫ ======
st.set_page_config(page_title="🇦🇩 Catalan Text Generator", page_icon="🇦🇩", layout="wide")
st.title("🇦🇩 Catalan Text Generator (CEFR)")

# ====== САЙДБАР: API КЛЮЧ И СЛОВАРЬ ======
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # API ключ — из secrets или ввод вручную
    try:
        default_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        default_key = ""
    
    api_key = st.text_input(
        "🔑 Groq API Key",
        value=default_key,
        type="password",
        help="Получить бесплатно: https://console.groq.com"
    )
    
    st.divider()
    st.subheader(f"📚 Словарь: {len(st.session_state.vocab)} слов")
    st.write(f"🔴 К повторению сегодня: **{count_due()}**")
    
    st.divider()
    st.subheader("💾 Экспорт / Импорт")
    if st.session_state.vocab:
        json_data = json.dumps(st.session_state.vocab, ensure_ascii=False, indent=2)
        st.download_button(
            "📥 Скачать словарь (JSON)",
            data=json_data,
            file_name="catalan_words.json",
            mime="application/json",
        )
    
    uploaded = st.file_uploader("📤 Загрузить словарь", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, list):
                st.session_state.vocab = data
                st.success(f"✅ Загружено слов: {len(data)}")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    
    st.divider()
    if st.button("🗑 Очистить словарь", use_container_width=True):
        st.session_state.vocab = []
        st.rerun()

if not api_key:
    st.warning("⚠️ Введите Groq API Key в сайдбаре слева. [Получить бесплатно](https://console.groq.com)")
    st.stop()

# ====== ВКЛАДКИ ======
tab1, tab2, tab3, tab4 = st.tabs(["✨ Генерация текста", "🔍 Быстрый перевод", "📚 Мои слова", "🎴 Карточки"])

# ==================== ВКЛАДКА 1: ГЕНЕРАЦИЯ ====================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("📝 Тема текста", value="els meus hobbies", placeholder="Например: un viatge a Barcelona")
        model = st.selectbox(
            "🤖 Модель",
            options=AVAILABLE_MODELS,
            format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x),
            index=0,
        )
    with col2:
        level = st.selectbox("📊 Уровень CEFR", CEFR_LEVELS, index=0)
        col_a, col_b = st.columns(2)
        with col_a:
            include_vocab = st.checkbox("Добавить словарь", value=True)
        with col_b:
            include_translation = st.checkbox("Перевод на русский", value=False)
    
    if st.button("✨ Сгенерировать текст", type="primary", use_container_width=True):
        if not topic:
            st.error("Введите тему!")
        else:
            with st.spinner(f"Генерирую текст ({model}, {level})..."):
                try:
                    result = generate_text(api_key, model, topic, level, include_vocab, include_translation)
                    st.session_state.last_text = result
                    st.session_state.last_parsed_words = parse_vocabulary(result)
                    st.success("✅ Текст сгенерирован!")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
    
    if st.session_state.last_text:
        st.markdown("### 📖 Сгенерированный текст")
        st.text_area("", value=st.session_state.last_text, height=300, label_visibility="collapsed")
        
        if st.session_state.last_parsed_words:
            n = len(st.session_state.last_parsed_words)
            if st.button(f"💾 Сохранить слова из текста ({n} шт.)", use_container_width=True):
                added = add_words_batch(st.session_state.last_parsed_words, level)
                skipped = n - added
                msg = f"✅ Добавлено новых: **{added}**"
                if skipped > 0:
                    msg += f" · ⚠️ Уже были (обновлены): {skipped}"
                st.success(msg)
                st.rerun()

# ==================== ВКЛАДКА 2: БЫСТРЫЙ ПЕРЕВОД ====================
with tab2:
    word_to_translate = st.text_input(
        "🔍 Слово или фраза на каталанском",
        placeholder="Например: malalt, anar, gràcies",
    )
    if st.button("🔍 Перевести", type="primary"):
        if not word_to_translate:
            st.warning("Введите слово!")
        else:
            with st.spinner("Перевожу..."):
                try:
                    result = translate_word(api_key, model, word_to_translate, st.session_state.last_text)
                    st.markdown("### 📖 Результат перевода")
                    st.markdown(result)
                    
                    pair = extract_translation_pair(result)
                    if pair:
                        w, t = pair
                        st.session_state["last_translated"] = (w, t)
                        if st.button(f"💾 Сохранить «{w} — {t}» в словарь"):
                            is_new = add_word(w, t, level)
                            if is_new:
                                st.success(f"✅ Добавлено: {w} — {t}")
                            else:
                                st.info(f"ℹ️ Обновлён перевод для: {w}")
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")

# ==================== ВКЛАДКА 3: МОИ СЛОВА ====================
with tab3:
    if not st.session_state.vocab:
        st.info("📭 Словарь пуст. Добавьте слова через генерацию текста или быстрый перевод.")
    else:
        st.write(f"**Всего слов:** {len(st.session_state.vocab)} · **К повторению сегодня:** {count_due()}")
        
        sorted_words = sorted(st.session_state.vocab, key=lambda w: w.get("added", ""), reverse=True)
        today = datetime.now().date()
        
        for i, w in enumerate(sorted_words):
            try:
                review_date = datetime.fromisoformat(w["next_review"]).date()
                is_due = review_date <= today
            except Exception:
                is_due = True
            
            col_w, col_t, col_lvl, col_del = st.columns([3, 3, 1, 1])
            with col_w:
                st.markdown(f"**{w['word']}**{' 🔴' if is_due else ''}")
            with col_t:
                st.write(w["translation"])
            with col_lvl:
                st.caption(w.get("level", "?"))
            with col_del:
                if st.button("🗑", key=f"del_{i}"):
                    remove_word(w["word"])
                    st.rerun()

# ==================== ВКЛАДКА 4: КАРТОЧКИ ====================
with tab4:
    due_words = get_due_words()
    
    if not st.session_state.in_session:
        if not due_words:
            if not st.session_state.vocab:
                st.info("📭 Словарь пуст. Сначала добавьте слова.")
            else:
                st.success(f"🎉 Всё выучено! Все {len(st.session_state.vocab)} слов повторены. Возвращайтесь завтра!")
        else:
            st.info(f"🔴 Готово к повторению: **{len(due_words)}** слов")
            if st.button(f"🎴 Начать сессию ({len(due_words)} карточек)", type="primary", use_container_width=True):
                st.session_state.in_session = True
                st.session_state.flashcard_index = 0
                st.session_state.flashcard_show_answer = False
                st.session_state.session_correct = 0
                st.session_state.session_incorrect = 0
                st.rerun()
    else:
        idx = st.session_state.flashcard_index
        
        if idx >= len(due_words):
            total = st.session_state.session_correct + st.session_state.session_incorrect
            st.success(f"🎉 Сессия завершена! Повторено: {total}")
            st.write(f"✅ Правильно: **{st.session_state.session_correct}**")
            st.write(f"❌ Неправильно: **{st.session_state.session_incorrect}**")
            if st.button("🏁 Завершить", use_container_width=True):
                st.session_state.in_session = False
                st.rerun()
        else:
            word_data = due_words[idx]
            st.progress((idx) / len(due_words))
            st.write(f"Карточка {idx + 1} из {len(due_words)} · ✅ {st.session_state.session_correct} · ❌ {st.session_state.session_incorrect}")
            
            st.markdown(
                f"""
                <div style="background-color: #fff8e1; border: 2px solid #ffc107; border-radius: 12px; padding: 40px; text-align: center;">
                    <div style="font-size: 42px; font-weight: bold; font-family: Consolas;">{word_data['word']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            if not st.session_state.flashcard_show_answer:
                if st.button("👁 Показать ответ", use_container_width=True):
                    st.session_state.flashcard_show_answer = True
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #e8f5e9; border: 2px solid #81c784; border-radius: 12px; padding: 25px; text-align: center; margin-top: 15px;">
                        <div style="font-size: 26px; color: #2e7d32; font-family: Consolas;">{word_data['translation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("❌ Не знал", use_container_width=True):
                        mark_incorrect(word_data["word"])
                        st.session_state.session_incorrect += 1
                        st.session_state.flashcard_index += 1
                        st.session_state.flashcard_show_answer = False
                        st.rerun()
                with col_b:
                    if st.button("✅ Знал", use_container_width=True):
                        mark_correct(word_data["word"])
                        st.session_state.session_correct += 1
                        st.session_state.flashcard_index += 1
                        st.session_state.flashcard_show_answer = False
                        st.rerun()
                
                if st.button("🏁 Завершить сессию"):
                    st.session_state.in_session = False
                    st.rerun()