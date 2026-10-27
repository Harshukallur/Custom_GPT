# app/rag.py
"""
Pure-Python Retrieval-Augmented Generation (RAG) helpers.

Pipeline:
    documents/*.txt|*.md  ->  chunker  ->  TF-IDF index  ->  cosine-similarity retriever

Only the standard library is used (no numpy / scikit-learn required).
"""

import math
import os
import re
import threading
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass

ALLOWED_EXTENSIONS = {".txt", ".md", ".docx"}

def _parse_docx(path):
    """Parse text content from a Microsoft Word .docx file in pure Python."""
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for paragraph in tree.iter("{" + namespaces["w"] + "}p"):
                texts = [node.text for node in paragraph.iter("{" + namespaces["w"] + "}t") if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to parse Word document: {e}")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its just me more most my
    no nor not of off on once only or other our out over own same she should so some such than
    that the their them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your""".split()
)


def _stem(token):
    """Very light suffix stripping so that 'modes' matches 'mode'."""
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    if token.endswith(("sses", "ches", "shes", "xes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text):
    """Lowercase, split into alphanumeric tokens, drop stopwords, lightly stem."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    """A piece of a document that is indexed and retrieved as one unit."""

    source: str
    index: int
    text: str


def chunk_text(text, chunk_size=120, overlap=30):
    """
    Split text into overlapping chunks of roughly ``chunk_size`` words.

    Sentences are kept intact where possible; sentences longer than ``chunk_size``
    are hard-split on word boundaries. The last ``overlap`` words of a chunk are
    carried into the next one to preserve context across boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    # Build a list of "units" (sentences, or word slices of over-long sentences).
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                units.append(words[i : i + chunk_size])

    chunks = []
    current = []
    for unit in units:
        if current and len(current) + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            # Overlap must never make the new chunk overflow on its own.
            if len(current) + len(unit) > chunk_size:
                current = []
        current.extend(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


class RAGEngine:
    """Loads documents from a folder, builds a TF-IDF index, and retrieves chunks."""

    def __init__(self, documents_dir, chunk_size=120, overlap=30):
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._lock = threading.RLock()
        self._chunks = []
        self._postings = defaultdict(list)  # term -> [(chunk_id, weight)]
        self._idf = {}
        self._signature = None
        os.makedirs(self.documents_dir, exist_ok=True)
        self.rebuild()

    # ------------------------------------------------------------------ files
    def _document_files(self):
        names = []
        for name in sorted(os.listdir(self.documents_dir)):
            path = os.path.join(self.documents_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS:
                names.append(name)
        return names

    def _compute_signature(self):
        sig = []
        for name in self._document_files():
            st = os.stat(os.path.join(self.documents_dir, name))
            sig.append((name, st.st_mtime_ns, st.st_size))
        return tuple(sig)

    # ------------------------------------------------------------------ index
    def rebuild(self):
        """(Re)build the whole index from the documents folder."""
        with self._lock:
            chunks = []
            for name in self._document_files():
                path = os.path.join(self.documents_dir, name)
                ext = os.path.splitext(name)[1].lower()
                text = ""
                try:
                    if ext == ".docx":
                        text = _parse_docx(path)
                    else:
                        with open(path, "r", encoding="utf-8") as fh:
                            text = fh.read()
                except (OSError, UnicodeDecodeError, ValueError):
                    continue  # skip unreadable or corrupt files
                for i, piece in enumerate(chunk_text(text, self.chunk_size, self.overlap)):
                    chunks.append(Chunk(source=name, index=i, text=piece))

            term_counts = [Counter(tokenize(c.text)) for c in chunks]
            n_docs = len(chunks)
            df = Counter()
            for counts in term_counts:
                df.update(counts.keys())
            idf = {t: math.log((1 + n_docs) / (1 + d)) + 1.0 for t, d in df.items()}

            postings = defaultdict(list)
            for chunk_id, counts in enumerate(term_counts):
                weights = {t: (1.0 + math.log(c)) * idf[t] for t, c in counts.items()}
                norm = math.sqrt(sum(w * w for w in weights.values()))
                if norm == 0:
                    continue
                for t, w in weights.items():
                    postings[t].append((chunk_id, w / norm))

            self._chunks = chunks
            self._idf = idf
            self._postings = postings
            self._signature = self._compute_signature()

    def _refresh_if_stale(self):
        with self._lock:
            if self._compute_signature() != self._signature:
                self.rebuild()

    # -------------------------------------------------------------- retrieval
    def retrieve(self, query, top_k=3, min_score=0.1):
        """
        Return up to ``top_k`` chunks most similar to ``query`` (cosine similarity
        on TF-IDF vectors) as dicts: source, chunk_index, score, text.
        """
        self._refresh_if_stale()
        with self._lock:
            counts = Counter(t for t in tokenize(query) if t in self._idf)
            if not counts:
                return []
            q_weights = {t: (1.0 + math.log(c)) * self._idf[t] for t, c in counts.items()}
            q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

            scores = defaultdict(float)
            for t, qw in q_weights.items():
                for chunk_id, dw in self._postings.get(t, ()):
                    scores[chunk_id] += (qw / q_norm) * dw

            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            results = []
            for chunk_id, score in ranked[:top_k]:
                if score < min_score:
                    break
                c = self._chunks[chunk_id]
                results.append(
                    {"source": c.source, "chunk_index": c.index, "score": round(score, 4), "text": c.text}
                )
            return results

    # ------------------------------------------------------ query introspection
    def get_query_vector(self, query, limit=None):
        """
        Return the TF-IDF query vector terms, highest weight first, as dicts:
        term, count, idf, weight (raw tf-idf) and normalized (L2-normalized weight).
        Terms absent from the index are ignored, matching what ``retrieve`` uses.
        """
        self._refresh_if_stale()
        with self._lock:
            counts = Counter(t for t in tokenize(query or "") if t in self._idf)
            if not counts:
                return []
            weights = {t: (1.0 + math.log(c)) * self._idf[t] for t, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values()))
            terms = [
                {
                    "term": t,
                    "count": counts[t],
                    "idf": round(self._idf[t], 4),
                    "weight": round(w, 4),
                    "normalized": round(w / norm, 4),
                }
                for t, w in weights.items()
            ]
        terms.sort(key=lambda d: (-d["weight"], d["term"]))
        return terms[:limit] if limit else terms

    def extract_keywords(self, query, top_n=5):
        """Return the ``top_n`` highest-weighted query terms as (term, normalized_weight) tuples."""
        return [(d["term"], d["normalized"]) for d in self.get_query_vector(query, limit=top_n)]

    # -------------------------------------------------------------- documents
    def get_document_chunks(self, name):
        """Return the chunk texts of a document, in order (empty list if unknown)."""
        self._refresh_if_stale()
        with self._lock:
            return [c.text for c in self._chunks if c.source == name]

    def list_documents(self):
        """Return metadata for every indexed document."""
        self._refresh_if_stale()
        with self._lock:
            chunk_counts = Counter(c.source for c in self._chunks)
            docs = []
            for name in self._document_files():
                docs.append(
                    {
                        "name": name,
                        "size": os.path.getsize(os.path.join(self.documents_dir, name)),
                        "chunks": chunk_counts.get(name, 0),
                    }
                )
            return docs

    def save_document(self, filename, data):
        """
        Save raw bytes as a document and re-index.

        :raises ValueError: for a bad extension, empty/non-UTF-8 content, or invalid name.
        """
        name = os.path.basename(filename or "")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
        ext = os.path.splitext(name)[1].lower()
        if not name or ext not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file type")
        
        if ext == ".docx":
            if len(data) == 0:
                raise ValueError("File is empty")
            with self._lock:
                save_path = os.path.join(self.documents_dir, name)
                with open(save_path, "wb") as fh:
                    fh.write(data)
                try:
                    # Validate that docx is parseable
                    _parse_docx(save_path)
                except Exception as e:
                    try:
                        os.remove(save_path)
                    except OSError:
                        pass
                    raise ValueError(f"Corrupt or invalid Word document: {e}")
                self.rebuild()
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("File must be UTF-8 text")
            if not text.strip():
                raise ValueError("File is empty")
            with self._lock:
                with open(os.path.join(self.documents_dir, name), "w", encoding="utf-8") as fh:
                    fh.write(text)
                self.rebuild()
        return name
