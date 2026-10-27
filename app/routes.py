# app/routes.py

from flask import Blueprint, render_template, request, jsonify
import os
import re
from app.utils import format_response
from app.chat import ChatManager
from app.rag import RAGEngine
from logger import CustomLogger

# Create a Blueprint for the main application
main = Blueprint('main', __name__)

# Initialize the ChatManager
chat_manager = ChatManager()  # Ensure this class is defined in app/chat.py

# Initialize the RAG engine over the project-level documents/ folder
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'documents')
MAX_UPLOAD_BYTES = 1024 * 1024  # 1 MB
MAX_LLM_CONTEXT_CHARS = 12000  # cap on document text sent to the LLM for summaries/suggestions
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.1
MAX_TOP_K = 20
rag_engine = RAGEngine(DOCUMENTS_DIR)
logger = CustomLogger().get_logger()

def _parse_retrieval_params(payload):
    """Validate optional top_k / min_score from a request payload.

    :return: (top_k, min_score) on success, or raises ValueError with a safe message.
    """
    top_k = payload.get('top_k', DEFAULT_TOP_K)
    min_score = payload.get('min_score', DEFAULT_MIN_SCORE)
    if isinstance(top_k, bool) or not isinstance(top_k, (int, float)) or int(top_k) != top_k:
        raise ValueError("top_k must be an integer")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError("min_score must be a number")
    top_k = int(top_k)
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    return top_k, float(min_score)


def _document_context(payload):
    """Resolve the 'document' name in a payload to truncated chunk text.

    :return: (context_text, None) on success, or (None, (error_response, status)).
    """
    name = payload.get('document')
    if not name or not isinstance(name, str):
        return None, (jsonify({"error": "No document provided"}), 400)
    chunks = rag_engine.get_document_chunks(name)
    if not chunks:
        return None, (jsonify({"error": "Document not found"}), 404)
    return "\n\n".join(chunks)[:MAX_LLM_CONTEXT_CHARS], None


@main.route('/')
def index():
    """Render the chat interface."""
    return render_template('chat.html')  # Renders the chat.html template

@main.route('/test')
def test():
    return render_template('base.html')  # Assuming base.html exists in templates/

@main.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from users."""
    payload = request.get_json(silent=True) or {}
    user_message = payload.get('message')
    
    if not user_message or not isinstance(user_message, str):
        return jsonify({"error": "No message provided"}), 400

    try:
        top_k, min_score = _parse_retrieval_params(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Retrieve relevant document chunks (RAG) based on user's core message
    sources = rag_engine.retrieve(user_message, top_k=top_k, min_score=min_score)
    query_vector = rag_engine.get_query_vector(user_message)

    # Process optional inline base64 attachment (image/document)
    attachment = payload.get('attachment')
    attachment_text = ""
    if attachment and isinstance(attachment, dict):
        filename = attachment.get('filename')
        b64_data = attachment.get('data')
        if filename and b64_data:
            import base64
            import io
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            try:
                file_bytes = base64.b64decode(b64_data)
            except Exception:
                return jsonify({"error": "Invalid base64 attachment data"}), 400

            ext = os.path.splitext(filename)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                from PIL import Image
                try:
                    with Image.open(io.BytesIO(file_bytes)) as img:
                        width, height = img.size
                        fmt = img.format
                        mode = img.mode
                        # Simple dominant color estimation
                        img_small = img.resize((50, 50))
                        colors = img_small.getcolors(2500)
                        dominant_info = ""
                        if colors:
                            colors.sort(key=lambda c: -c[0])
                            hex_colors = []
                            for count, color in colors[:3]:
                                if isinstance(color, tuple):
                                    hex_colors.append("#" + "".join(f"{x:02x}" for x in color[:3]))
                            dominant_info = f", dominant colors: {', '.join(hex_colors)}"
                        attachment_text = f"[Attached Image: {filename} ({width}x{height} {fmt}, Mode: {mode}{dominant_info})]"
                except Exception as e:
                    attachment_text = f"[Attached Image: {filename} (Error analyzing file: {e})]"
            elif ext == ".docx":
                try:
                    import zipfile
                    import xml.etree.ElementTree as ET
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
                        xml_content = docx.read("word/document.xml")
                        tree = ET.fromstring(xml_content)
                        namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                        paragraphs = []
                        for paragraph in tree.iter("{" + namespaces["w"] + "}p"):
                            texts = [node.text for node in paragraph.iter("{" + namespaces["w"] + "}t") if node.text]
                            if texts:
                                paragraphs.append("".join(texts))
                        attachment_text = f"[Attached Document: {filename}]\n" + "\n".join(paragraphs)
                except Exception as e:
                    attachment_text = f"[Attached Document: {filename} (Error parsing file: {e})]"
            elif ext in (".txt", ".md"):
                try:
                    attachment_text = f"[Attached Document: {filename}]\n" + file_bytes.decode("utf-8")
                except Exception as e:
                    attachment_text = f"[Attached Document: {filename} (Error decoding file: {e})]"
            else:
                return jsonify({"error": "Unsupported attachment type"}), 400

    # Prepend attachment text if present to ground prompt contextualization
    full_prompt = user_message
    if attachment_text:
        full_prompt = f"{attachment_text}\n\n-------------------------\n{user_message}"

    # Get AI response using updated ChatManager logic
    ai_response = chat_manager.get_response(full_prompt, context_chunks=sources)

    # Format the response if necessary (e.g., replace newlines)
    formatted_response = format_response(ai_response)

    # Return the formatted AI's response plus the chunks that were used as context
    return jsonify({"response": formatted_response, "sources": sources, "query_vector": query_vector})

@main.route('/api/query-vector', methods=['POST'])
def query_vector_endpoint():
    """Return the TF-IDF query vector terms for a query (for the developer console)."""
    payload = request.get_json(silent=True) or {}
    query = payload.get('query')
    if not isinstance(query, str):
        return jsonify({"error": "No query provided"}), 400
    return jsonify({"query_vector": rag_engine.get_query_vector(query)})

@main.route('/api/summarize', methods=['POST'])
def summarize():
    """Summarize a document using the LLM."""
    payload = request.get_json(silent=True) or {}
    context, error = _document_context(payload)
    if error:
        return error
    prompt = (
        "Write a concise summary (at most 5 sentences) of the following document. "
        "Use only the document's content.\n\n"
        f"Document:\n{context}"
    )
    summary = chat_manager.client.get_response([{"role": "user", "content": prompt}])
    return jsonify({"document": payload['document'], "summary": summary})

@main.route('/api/suggest-questions', methods=['POST'])
def suggest_questions():
    """Generate 3 suggested questions about a document using the LLM."""
    payload = request.get_json(silent=True) or {}
    context, error = _document_context(payload)
    if error:
        return error
    prompt = (
        "Based on the following document, write exactly 3 high-quality questions a reader "
        "could ask about it. Each must be answerable from the document. "
        "Output only the 3 questions, one per line, with no numbering or extra text.\n\n"
        f"Document:\n{context}"
    )
    raw = chat_manager.client.get_response([{"role": "user", "content": prompt}])
    questions = []
    for line in raw.splitlines():
        line = re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s*', '', line).strip()
        if line.endswith('?'):
            questions.append(line)
    return jsonify({"document": payload['document'], "questions": questions[:3]})

@main.route('/api/documents', methods=['GET'])
def list_documents():
    """List the documents available for retrieval."""
    return jsonify({"documents": rag_engine.list_documents()})

@main.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload a .txt or .md document and add it to the index."""
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    data = file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File is too large (max 1 MB)"}), 413

    try:
        name = rag_engine.save_document(file.filename, data)
    except ValueError as e:
        # Messages raised by save_document are safe, user-facing validation errors
        return jsonify({"error": str(e)}), 400
    except OSError:
        logger.error("Failed to save uploaded document", exc_info=True)
        return jsonify({"error": "Could not save the file"}), 500

    logger.info("Document uploaded and indexed", extra={"document": name})
    return jsonify({"message": "Uploaded", "name": name, "documents": rag_engine.list_documents()})

@main.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history."""
    chat_manager.conversation_history.clear()
    return jsonify({"message": "Chat history cleared"})

def format_response(response):
    # Replace newlines with <br> for line breaks
    response = response.replace('\n', '<br>')
    
    # Convert bold text (e.g., **text**) to <strong> tags
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Convert italic text (e.g., *text*) to <em> tags
    response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', response)

    # Detect headings (e.g., "## Heading" or "# Heading")
    response = re.sub(r'(?m)^(#+)\s*(.*)', lambda m: f'<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>', response)

    # Handle lists (e.g., "1. Item" or "- Item")
    response = re.sub(r'(?m)^\d+\.\s*(.*)', r'<li>\1</li>', response)
    response = re.sub(r'(?m)^\-\s*(.*)', r'<li>\1</li>', response)
    
    # Wrap list items in <ul>
    if '<li>' in response:
        response = '<ul>' + response + '</ul>'

    # Handle code blocks (using triple backticks)
    if "```" in response:
        parts = response.split("```")
        formatted_response = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd index means it's code
                formatted_response += f'<pre class="code-block"><code>{part}</code></pre>'
            else:
                formatted_response += part.replace('\n', '<br>')  # Replace newlines with <br>
        return formatted_response

    return response