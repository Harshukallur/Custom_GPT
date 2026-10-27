# tests/test_rag.py
"""Tests for the RAG pipeline. Run from custom_gpt/:  python -m unittest discover -s tests -v"""

import io
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag import RAGEngine, chunk_text, tokenize  # noqa: E402


class ChunkerTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])

    def test_chunks_respect_size_and_overlap(self):
        text = " ".join(f"word{i}." for i in range(100))
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 20)
        # last words of chunk N appear at the start of chunk N+1
        self.assertEqual(chunks[0].split()[-5:], chunks[1].split()[:5])

    def test_long_sentence_is_hard_split(self):
        chunks = chunk_text(" ".join(["x"] * 50), chunk_size=10, overlap=0)
        self.assertEqual(len(chunks), 5)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)

    def test_tokenize_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("The modes are running"), ["mode", "runn"])
        self.assertEqual(tokenize("mode modes boxes"), ["mode", "mode", "box"])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.write("cats.txt", "Cats are small carnivorous mammals. Cats purr and chase mice.")
        self.write("space.md", "Rockets carry satellites into orbit around the Earth.")
        self.engine = RAGEngine(self.tmp.name)

    def write(self, name, text):
        with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_retrieves_relevant_document(self):
        res = self.engine.retrieve("Why do cats purr?")
        self.assertEqual(res[0]["source"], "cats.txt")
        res = self.engine.retrieve("rocket orbit")
        self.assertEqual(res[0]["source"], "space.md")

    def test_no_match_returns_empty(self):
        self.assertEqual(self.engine.retrieve("quantum chromodynamics"), [])
        self.assertEqual(self.engine.retrieve("the and of"), [])

    def test_list_documents(self):
        names = [d["name"] for d in self.engine.list_documents()]
        self.assertEqual(names, ["cats.txt", "space.md"])

    def test_save_document_reindexes(self):
        self.engine.save_document("dogs.txt", b"Dogs bark and fetch sticks.")
        self.assertEqual(self.engine.retrieve("dogs fetch")[0]["source"], "dogs.txt")

    def test_save_document_validation(self):
        for name, data in [("x.exe", b"hi"), ("x.txt", b"   "), ("x.txt", b"\xff\xfe\x00")]:
            with self.assertRaises(ValueError):
                self.engine.save_document(name, data)

    def test_filename_is_sanitised(self):
        name = self.engine.save_document("../../etc/evil.txt", b"hello world")
        self.assertEqual(name, "evil.txt")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "evil.txt")))

    def test_external_file_change_detected(self):
        self.write("new.txt", "Zebras have black and white stripes.")
        self.assertEqual(self.engine.retrieve("zebra stripes")[0]["source"], "new.txt")

    def test_docx_parsing_and_indexing(self):
        import zipfile
        import io
        
        # Open XML structure for a simple document with paragraphs
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            '<w:p><w:r><w:t>This is a paragraph from a Word document.</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>It contains multiple sentences. Here is the second sentence.</w:t></w:r></w:p>'
            '</w:body>'
            '</w:document>'
        )
        
        docx_data = io.BytesIO()
        with zipfile.ZipFile(docx_data, "w") as z:
            z.writestr("word/document.xml", xml_content)
        
        # Save document
        name = self.engine.save_document("test_doc.docx", docx_data.getvalue())
        self.assertEqual(name, "test_doc.docx")
        
        # Try retrieving content
        res = self.engine.retrieve("Word document paragraph")
        self.assertEqual(res[0]["source"], "test_doc.docx")
        self.assertIn("This is a paragraph", res[0]["text"])


    def test_query_vector_and_keywords(self):
        vec = self.engine.get_query_vector("cats purr cats unknownterm")
        self.assertEqual({v["term"] for v in vec}, {"cat", "purr"})
        self.assertEqual(vec[0]["term"], "cat")  # repeated term weighs more
        self.assertAlmostEqual(sum(v["normalized"] ** 2 for v in vec), 1.0, places=2)
        self.assertEqual(self.engine.extract_keywords("cats purr", top_n=1)[0][0] in {"cat", "purr"}, True)
        self.assertEqual(len(self.engine.get_query_vector("cats purr", limit=1)), 1)
        self.assertEqual(self.engine.get_query_vector("the and of"), [])

    def test_retrieve_respects_top_k_and_min_score(self):
        self.write("more_cats.txt", "Cats sleep a lot.")
        self.assertEqual(len(self.engine.retrieve("cats", top_k=1)), 1)
        self.assertEqual(len(self.engine.retrieve("cats", top_k=5)), 2)
        self.assertEqual(self.engine.retrieve("cats", top_k=5, min_score=0.999), [])

    def test_get_document_chunks(self):
        self.assertEqual(len(self.engine.get_document_chunks("cats.txt")), 1)
        self.assertEqual(self.engine.get_document_chunks("missing.txt"), [])


class RoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ.setdefault("API_KEY", "test-key")
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # config/ is relative
        from app import create_app
        import app.routes as routes

        cls.routes = routes
        routes.rag_engine = RAGEngine(cls.tmp.name)
        cls.app = create_app()
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_upload_list_and_chat(self):
        r = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"Pandas eat bamboo all day long."), "pandas.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 200)

        r = self.client.get("/api/documents")
        self.assertIn("pandas.txt", [d["name"] for d in r.get_json()["documents"]])

        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value="Bamboo.") as m:
            r = self.client.post("/api/chat", json={"message": "What do pandas eat?"})
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(body["sources"][0]["source"], "pandas.txt")
        prompt = m.call_args[0][0][-1]["content"]
        self.assertIn("Pandas eat bamboo", prompt)

    def test_upload_rejects_bad_input(self):
        self.assertEqual(self.client.post("/api/upload").status_code, 400)
        r = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"x"), "bad.exe")},
            content_type="multipart/form-data",
        )
        self.assertEqual(r.status_code, 400)

    def test_chat_requires_message(self):
        self.assertEqual(self.client.post("/api/chat", json={}).status_code, 400)

    def test_chat_with_attachments(self):
        import base64
        # Test document attachment (txt)
        txt_b64 = base64.b64encode(b"This is the attached document text.").decode("utf-8")
        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value="Parsed.") as m:
            r = self.client.post(
                "/api/chat",
                json={
                    "message": "Verify attachment",
                    "attachment": {"filename": "attached.txt", "data": txt_b64}
                }
            )
            self.assertEqual(r.status_code, 200)
            prompt = m.call_args[0][0][-1]["content"]
            self.assertIn("This is the attached document text.", prompt)

        # Test image attachment (1x1 transparent GIF)
        img_b64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value="Parsed.") as m:
            r = self.client.post(
                "/api/chat",
                json={
                    "message": "Verify image",
                    "attachment": {"filename": "dot.gif", "data": img_b64}
                }
            )
            self.assertEqual(r.status_code, 200)
            prompt = m.call_args[0][0][-1]["content"]
            self.assertIn("Attached Image: dot.gif", prompt)

    def test_chat_custom_params_and_validation(self):
        self.routes.rag_engine.save_document("koalas.txt", b"Koalas eat eucalyptus leaves.")
        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value="ok"):
            r = self.client.post("/api/chat", json={"message": "koalas eucalyptus", "top_k": 1, "min_score": 0.99})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.get_json()["sources"], [])
            self.assertEqual(r.get_json()["query_vector"][0]["term"] in {"koala", "eucalyptu", "eucalyptus"}, True)
            for bad in ({"top_k": 0}, {"top_k": 999}, {"top_k": "a"}, {"min_score": 2}, {"min_score": "x"}):
                r = self.client.post("/api/chat", json={"message": "koalas", **bad})
                self.assertEqual(r.status_code, 400, bad)

    def test_query_vector_endpoint(self):
        self.routes.rag_engine.save_document("otters.txt", b"Otters hold hands while sleeping.")
        r = self.client.post("/api/query-vector", json={"query": "otters sleeping"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["query_vector"]), 2)
        self.assertEqual(self.client.post("/api/query-vector", json={}).status_code, 400)

    def test_summarize_endpoint(self):
        self.routes.rag_engine.save_document("owls.txt", b"Owls hunt at night.")
        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value="Owls are nocturnal.") as m:
            r = self.client.post("/api/summarize", json={"document": "owls.txt"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["summary"], "Owls are nocturnal.")
        self.assertIn("Owls hunt at night", m.call_args[0][0][0]["content"])
        self.assertEqual(self.client.post("/api/summarize", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/summarize", json={"document": "nope.txt"}).status_code, 404)

    def test_suggest_questions_endpoint(self):
        self.routes.rag_engine.save_document("bats.txt", b"Bats use echolocation to navigate.")
        llm = "1. What do bats use to navigate?\n- How do bats find food?\nNot a question\n* Where do bats live?\nExtra one?"
        with mock.patch.object(self.routes.chat_manager.client, "get_response", return_value=llm):
            r = self.client.post("/api/suggest-questions", json={"document": "bats.txt"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.get_json()["questions"],
            ["What do bats use to navigate?", "How do bats find food?", "Where do bats live?"],
        )
        self.assertEqual(self.client.post("/api/suggest-questions", json={"document": "nope.txt"}).status_code, 404)

    def test_clear_chat_endpoint(self):
        self.routes.chat_manager.add_message("user", "Hello")
        self.routes.chat_manager.add_message("assistant", "Hi there")
        self.assertGreaterEqual(len(self.routes.chat_manager.conversation_history), 2)
        
        r = self.client.post("/api/chat/clear")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["message"], "Chat history cleared")
        self.assertEqual(len(self.routes.chat_manager.conversation_history), 0)


if __name__ == "__main__":
    unittest.main()
