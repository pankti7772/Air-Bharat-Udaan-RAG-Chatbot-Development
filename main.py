# ===================== LOAD ENV =====================

from dotenv import load_dotenv
load_dotenv()

import os
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

import re
from datetime import datetime
from typing import TypedDict
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    UserMixin,
    current_user
)
from flask_dance.contrib.google import make_google_blueprint, google
from pypdf import PdfReader

from langchain_groq import ChatGroq
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END


# ===================== CONFIG =====================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_REGION = os.getenv("AWS_REGION")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL_ID")

DATA_DIR = "attached_assets"
FAISS_PATH = "faiss_index"

app = Flask(__name__, template_folder="templates", static_folder="static")

CORS(app, supports_credentials=True)

app.config["SECRET_KEY"] = "super_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///udan_users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["GOOGLE_OAUTH_CLIENT_ID"] = os.getenv("GOOGLE_CLIENT_ID")
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = os.getenv("GOOGLE_CLIENT_SECRET")

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "google.login"


# ===================== UNAUTHORIZED =====================

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify({"answer": "🔒 Please login first to use this feature."}), 401
    return redirect(url_for("google.login"))


# ===================== GOOGLE OAUTH =====================

google_bp = make_google_blueprint(
    scope=["openid", "email", "profile"],
    redirect_to="google_login"
)

app.register_blueprint(google_bp, url_prefix="/login")


# ===================== DATABASE =====================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(200), unique=True)
    email = db.Column(db.String(200))
    name = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class LoginHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(100))


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ===================== MODEL FACTORY =====================

def get_llm(provider: str, model_name: str):

    if provider == "groq":
        return ChatGroq(
            model=model_name,
            groq_api_key=GROQ_API_KEY,
            temperature=0.2,
            max_tokens=800
        )

    elif provider == "bedrock":
        return ChatBedrock(
            model_id=model_name,
            region_name=AWS_REGION,
            model_kwargs={"temperature": 0.2, "max_tokens": 800}
        )

    else:
        raise ValueError("Invalid provider")


# ===================== EMBEDDINGS =====================

try:
    embeddings = BedrockEmbeddings(
        model_id=EMBED_MODEL,
        region_name=AWS_REGION
    )
except Exception as e:
    logger.warning(f"⚠️  Bedrock embeddings initialization failed: {e}")
    embeddings = None


# ===================== PROMPTS =====================

FACT_PROMPT = PromptTemplate.from_template("""
Use ONLY the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY what is asked.
- If partial information exists, answer with available details.
- Do NOT say "Not found" if some relevant info exists.
- Only say "Not found in documents" if absolutely no info exists.

Answer:
""")

COMPARE_PROMPT = PromptTemplate.from_template("""
You are comparing entities using ONLY the provided context.

Context:
{context}

Question:
{question}

Provide a clear comparison using bullet points.
Do NOT use table format.
Separate each entity with headings.
Use simple language suitable for travelers.
If information is missing say "Not found in documents".

Answer:
""")


# ===================== VECTOR STORE =====================

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def load_documents():

    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    if not os.path.exists(DATA_DIR):
        return docs

    for file in os.listdir(DATA_DIR):

        path = os.path.join(DATA_DIR, file)

        if file.endswith(".txt"):
            raw = open(path, encoding="utf-8").read()

        elif file.endswith(".pdf"):
            raw = read_pdf(path)

        else:
            continue

        chunks = splitter.split_text(raw)

        for c in chunks:
            docs.append(Document(page_content=c))

    return docs


def build_or_load_vectorstore():

    if embeddings is None:
        logger.warning("Embeddings not available, skipping vectorstore load")
        return None

    if os.path.exists(FAISS_PATH):
        try:
            return FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            return None

    docs = load_documents()

    if not docs:
        logger.warning("No documents loaded, skipping vectorstore creation")
        return None

    try:
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(FAISS_PATH)
        return store
    except Exception as e:
        logger.warning(f"Failed to create FAISS store: {e}")
        return None


try:
    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20}
    ) if vectorstore else None
except Exception as e:
    logger.warning(f"⚠️  Vector store initialization failed: {e}")
    vectorstore = None
    retriever = None


# ===================== RAG =====================

class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    llm: object


def retrieve_node(state: RAGState):

    if not retriever:
        return {"context": ""}

    question = state["question"].lower()

    if any(x in question for x in ["all", "list", "show", "give"]):

        if "air india" in question:
            docs = retriever.vectorstore.similarity_search(
                "Air India flight delay", k=40
            )

        elif "indigo" in question:
            docs = retriever.vectorstore.similarity_search(
                "IndiGo flight delay", k=40
            )

        else:
            docs = retriever.vectorstore.similarity_search(
                question, k=40
            )

    else:
        docs = retriever.vectorstore.similarity_search(
            question, k=40
        )

    # ================= FILTER LOGIC (NEW) =================

    filtered = []

    for d in docs:
        text = d.page_content.lower()

        if "domestic" in question:
            # domestic flights → AI-xxx format
            if "ai-" in text:
                filtered.append(d.page_content)

        elif "international" in question:
            # international routes → no AI-xxx, city-to-city format
            if " to " in text and "ai-" not in text:
                filtered.append(d.page_content)

        else:
            filtered.append(d.page_content)

    # fallback if nothing matched
    if not filtered:
        filtered = [d.page_content for d in docs]

    context = "\n".join(filtered[:12])   # limit to 12 chunks

    return {"context": context}


def fact_node(state: RAGState):

    if not state["context"].strip():
        try:
            response = state["llm"].invoke(state["question"]).content.strip()
            return {"answer": response}
        except Exception as e:
            logger.error("FACT NODE ERROR:", exc_info=True)
            return {"answer": "Model temporarily unavailable."}

    q_lower = state["question"].lower()

    if any(word in q_lower for word in ["compare", "difference", "vs", "between"]):
        prompt = COMPARE_PROMPT
    else:
        prompt = FACT_PROMPT

    response = state["llm"].invoke(
        prompt.format(
            context=state["context"],
            question=state["question"]
        )
    ).content.strip()

    return {"answer": response}


graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("fact", fact_node)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "fact")
graph.add_edge("fact", END)

rag_graph = graph.compile()


# ===================== ROUTES =====================

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def home():
    try:
        user_name = current_user.name if current_user.is_authenticated else None
        return render_template("index.html", logged_in=current_user.is_authenticated, user_name=user_name)
    except Exception as e:
        logger.error(f"Home route error: {e}", exc_info=True)
        # Fallback if template not found
        return jsonify({"error": "Template not found", "status": "ok"}), 500


@app.route("/google_login")
def google_login():

    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    info = resp.json()

    google_id = info.get("id")
    email = info.get("email")
    name = info.get("name")

    if not google_id:
        return "Login failed", 400

    user = User.query.filter_by(google_id=google_id).first()

    if not user:
        user = User(google_id=google_id, email=email, name=name)
        db.session.add(user)
        db.session.commit()

    login_user(user)

    db.session.add(LoginHistory(
        user_id=user.id,
        ip_address=request.remote_addr
    ))

    db.session.commit()

    return redirect("/")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


@app.route("/api/history")
@login_required
def history():

    chats = Chat.query.filter_by(user_id=current_user.id).all()

    return jsonify([
        {
            "question": c.question,
            "answer": c.answer,
            "time": c.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for c in chats
    ])


@app.route("/api/query", methods=["POST"])
@login_required
def query():
    try:
        data = request.json

        original_q = re.sub(r"\s+", " ", data.get("question", "")).strip()
        q = original_q

        if not q:
            return jsonify({"answer": "Please ask a valid question."})

        q_lower = q.lower()

        # ===================== GENERAL MEMORY =====================
        last_question = session.get("last_question")
        last_answer = session.get("last_answer")

        follow_words = ["it", "this", "that", "those", "they", "above", "previous"]

        is_followup = False

        # detect follow-up ONLY by keyword when last_question exists
        if last_question and any(word in q_lower for word in follow_words):
            is_followup = True

        if is_followup:
            q = f"Previous Question: {last_question}\nPrevious Answer: {last_answer}\nCurrent Question: {original_q}"
        else:
            q = original_q
            last_question = None
            last_answer = None

        if "all questions" in q_lower:
            chats = Chat.query.filter_by(user_id=current_user.id).all()

            if not chats:
                return jsonify({"answer": "No questions asked yet."})

            question_list = "\n".join(
                f"{i+1}. {c.question}"
                for i, c in enumerate(chats)
            )

            return jsonify({"answer": question_list})

        if q_lower in ["hi", "hello", "hey"]:
            answer = "Hello! 😊 How can I assist you today?"
        else:
            provider = data.get("provider", "groq")
            model_name = data.get("model_name", "llama-3.1-8b-instant")

            logger.info(f"Using provider: {provider}, model: {model_name}")
            llm_instance = get_llm(provider, model_name)

            try:
                if retriever and rag_graph:
                    result = rag_graph.invoke({
                        "question": q,
                        "llm": llm_instance
                    })
                    answer = result["answer"]
                else:
                    logger.warning("Retriever or RAG graph not available, using direct LLM")
                    answer = llm_instance.invoke(original_q).content.strip()

            except Exception as e:
                logger.error(f"RAG ERROR: {e}", exc_info=True)

                # 🔁 fallback direct LLM
                try:
                    answer = llm_instance.invoke(original_q).content.strip()
                except Exception as e2:
                    logger.error(f"LLM ERROR: {e2}", exc_info=True)
                    answer = "Model temporarily unavailable."

        # store ALWAYS the latest clean question
        session["last_question"] = original_q
        session["last_answer"] = answer

        db.session.add(Chat(
            user_id=current_user.id,
            question=original_q,
            answer=answer
        ))

        db.session.commit()

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({"answer": f"Error: {str(e)}"}), 500


@app.errorhandler(500)
def handle_500(error):
    logger.error(f"500 Error: {error}", exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500


@app.errorhandler(Exception)
def handle_general_error(error):
    logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":

    with app.app_context():
        try:
            db.create_all()
            Chat.query.delete()
            db.session.commit()
            logger.info("✅ Chat history cleared on server restart.")
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)

    port = int(os.getenv("PORT", 8000))
    debug_mode = os.getenv("FLASK_ENV") == "development"
    
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    logger.info(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"Static folder: {app.static_folder}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=debug_mode)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise