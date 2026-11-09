# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The deployment verifier's verdict.

It used to pass anything that came back with a 200 and a non-empty body, so
NaN output ("!!!!") and loops ("the the the") verified, while a working
server built from the FastAPI recipe failed: the recipe answers under
"result", a key the verifier did not read.
"""

import http.server
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

import verify_inference as verify

GOOD_ANSWERS = [
    "A tensor is a multi-dimensional array of numbers, generalizing scalars, "
    "vectors and matrices to any number of dimensions.",
    "In machine learning, a tensor is simply an n-dimensional array. For "
    "example, an image is a 3-D tensor of height x width x channels.",
    "4",
    "Paris",
    "A tensor is a container for data, usually numbers. Tensors have a "
    "shape, a dtype, and a device -- e.g. `torch.zeros(2, 3)`.",
    "张量是一个多维数组，是标量、向量和矩阵在更高维度上的推广。",
    "Sure! Here's a short answer:\n\n- **Scalar**: 0-D\n- **Vector**: 1-D\n"
    "- **Matrix**: 2-D\n- **Tensor**: N-D",
]


class DegeneracyTest(unittest.TestCase):

    def test_normal_answers_pass(self):
        for text in GOOD_ANSWERS:
            with self.subTest(text=text):
                self.assertEqual(verify.degeneracy_failures(text), [])

    def test_degenerate_outputs_fail(self):
        cases = {
            "nan bangs": "!" * 64,
            "nan bangs spaced": "! " * 32,
            "word loop": "the " * 30,
            "phrase loop": "a tensor is " * 12,
            "unk": "<unk>" * 12,
            "pad mixed": "Hello <pad><pad><pad><pad><pad>",
            "replacement chars": "�" * 20,
            "empty": "   \n ",
        }
        for name, text in cases.items():
            with self.subTest(name):
                self.assertNotEqual(verify.degeneracy_failures(text), [])

    def test_short_emphasis_passes(self):
        self.assertEqual(verify.degeneracy_failures("Yes!!"), [])


class EnvelopeTest(unittest.TestCase):

    def test_fastapi_recipe_result_key(self):
        self.assertEqual(
            verify.extract_completion("custom_fastapi", {"result": "hi"}),
            "hi")

    def test_vllm_chat_and_completions(self):
        chat = {"choices": [{"message": {"content": "hi"}}]}
        plain = {"choices": [{"text": "hi"}]}
        self.assertEqual(verify.extract_completion("vllm", chat), "hi")
        self.assertEqual(verify.extract_completion("vllm", plain), "hi")
        self.assertEqual(verify.extract_completion("sglang", chat), "hi")
        self.assertEqual(verify.extract_completion("sglang", plain), "hi")

    def test_null_text_is_empty_not_a_crash(self):
        body = {"choices": [{"text": None}]}
        self.assertEqual(verify.extract_completion("vllm", body), "")
        self.assertEqual(verify.extract_completion("sglang", body), "")
        self.assertEqual(verify.strip_prompt_echo(None, "p"), "")

    def test_prompt_echo_is_stripped(self):
        prompt = verify.DEFAULT_PROMPT
        self.assertEqual(
            verify.strip_prompt_echo(prompt + " the the the", prompt),
            " the the the")
        self.assertEqual(verify.strip_prompt_echo("A tensor", prompt),
                         "A tensor")

    def test_requests_are_greedy(self):
        for engine, chat in (("vllm", True), ("vllm", False), ("sglang", True),
                             ("sglang", False), ("custom_fastapi", False)):
            _, payload = verify.build_request(engine, "m", "p", 8, chat)
            self.assertEqual(payload["temperature"], 0)


class _Handler(http.server.BaseHTTPRequestHandler):
    """Answers like the server under test; set by each test."""
    reply = staticmethod(lambda path, body: (200, {}))
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, body))
        status, payload = type(self).reply(self.path, body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class EndToEndTest(unittest.TestCase):

    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        _Handler.seen = []
        self.out = pathlib.Path(tempfile.mkdtemp()) / "verification.json"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def run_verifier(self, *extra):
        argv = [
            "verify_inference.py", "--port",
            str(self.port), "--output",
            str(self.out), *extra
        ]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            code = verify.main()
        return code, json.loads(self.out.read_text())

    def test_fastapi_recipe_answer_verifies(self):

        def reply(path, body):
            if "France" in body["prompt"]:
                return 200, {"result": "Paris."}
            return 200, {"result": GOOD_ANSWERS[0]}

        _Handler.reply = staticmethod(reply)
        code, report = self.run_verifier()
        self.assertEqual(code, 0)
        self.assertTrue(report["verified"])
        self.assertEqual(report["warnings"], [])
        self.assertTrue(report["known_answer"]["passed"])

    def test_nan_output_fails_despite_200(self):
        _Handler.reply = staticmethod(lambda p, b: (200, {"result": "!" * 64}))
        code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertFalse(report["verified"])
        self.assertTrue(report["failures"])
        self.assertIsNone(report["known_answer"])

    def test_echoed_prompt_does_not_hide_a_loop(self):
        _Handler.reply = staticmethod(lambda p, b:
                                      (200, {
                                          "result": b["prompt"] + " the" * 30
                                      }))
        code, report = self.run_verifier("--skip-known-answer")
        self.assertEqual(code, 1)

    def test_wrong_known_answer_is_only_a_warning(self):
        _Handler.reply = staticmethod(lambda p, b: (200, {
            "result": GOOD_ANSWERS[1]
        }))
        code, report = self.run_verifier()
        self.assertEqual(code, 0)
        self.assertTrue(report["verified"])
        self.assertFalse(report["known_answer"]["passed"])
        self.assertEqual(len(report["warnings"]), 1)

    def test_vllm_uses_chat_then_falls_back_without_template(self):

        def reply(path, body):
            if path == "/v1/chat/completions":
                return 400, {
                    "message":
                    "default chat template is no longer allowed, "
                    "so you must provide a chat template"
                }
            return 200, {"choices": [{"text": GOOD_ANSWERS[0]}]}

        _Handler.reply = staticmethod(reply)
        code, report = self.run_verifier("--engine", "vllm", "--model-id", "m",
                                         "--skip-known-answer")
        self.assertEqual(code, 0)
        self.assertEqual([p for p, _ in _Handler.seen],
                         ["/v1/chat/completions", "/v1/completions"])
        self.assertTrue(report["url"].endswith("/v1/completions"))

    def test_vllm_chat_answer_verifies(self):
        _Handler.reply = staticmethod(lambda p, b: (200, {
            "choices": [{
                "message": {
                    "content": "Paris"
                    if "France" in str(b) else GOOD_ANSWERS[0]
                }
            }]
        }))
        code, report = self.run_verifier("--engine", "vllm", "--model-id", "m")
        self.assertEqual(code, 0)
        self.assertEqual({p
                          for p, _ in _Handler.seen}, {"/v1/chat/completions"})
        self.assertTrue(report["known_answer"]["passed"])

    def test_sglang_chat_answer_verifies(self):
        _Handler.reply = staticmethod(lambda p, b: (200, {
            "choices": [{
                "message": {
                    "content": "Paris"
                    if "France" in str(b) else GOOD_ANSWERS[0]
                }
            }]
        }))
        code, report = self.run_verifier("--engine", "sglang", "--model-id",
                                         "m")
        self.assertEqual(code, 0)
        self.assertEqual(report["engine"], "sglang")
        self.assertEqual({p
                          for p, _ in _Handler.seen}, {"/v1/chat/completions"})
        self.assertTrue(report["known_answer"]["passed"])

    def test_null_vllm_text_fails_cleanly(self):
        _Handler.reply = staticmethod(lambda p, b: (200, {
            "choices": [{
                "text": None
            }]
        }) if p == "/v1/completions" else (400, {
            "message": "chat template"
        }))
        code, report = self.run_verifier("--engine", "vllm", "--model-id", "m")
        self.assertEqual(code, 1)
        self.assertEqual(report["failures"], ["empty completion"])

    def test_response_over_8kb_is_read_whole(self):
        padding = "x" * 20000
        _Handler.reply = staticmethod(lambda p, b: (
            200, {
                "result": "Paris"
                if "France" in b["prompt"] else GOOD_ANSWERS[0],
                "metadata": padding,
            }))
        code, report = self.run_verifier()
        self.assertEqual(code, 0)
        self.assertEqual(report["error"], "")

    def test_server_error_on_known_answer_fails(self):
        _Handler.reply = staticmethod(lambda p, b: (500, {
            "detail": "out of memory"
        }) if "France" in b["prompt"] else (200, {
            "result": GOOD_ANSWERS[0]
        }))
        code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertFalse(report["verified"])
        self.assertIn("HTTP 500", report["failures"][0])
        self.assertEqual(report["warnings"], [])

    def test_http_error_fails(self):
        _Handler.reply = staticmethod(lambda p, b: (500, {"detail": "boom"}))
        code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertEqual(report["failures"], ["HTTP status 500"])


if __name__ == "__main__":
    unittest.main()
