"""A tool-call tag quoted in a message is text, not the real special token."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import tokenizer as ST  # noqa: E402
from serve.frontend import literal_tags, mark_think_literals, unmark_think_literals  # noqa: E402
from serve.util.text import control_literals  # noqa: E402


class QuotedToolTags(unittest.TestCase):
    def setUp(self):
        b2u = ST.bytes_to_unicode()
        self.tool_tags = ["</tool_call>", "<tool_call>", "<tool_response>", "</tool_response>"]
        self.tok = ST.Tokenizer([b2u[b] for b in range(256)] + self.tool_tags + ["<|im_end|>"], [],
                                [1] * 256 + [4] * len(self.tool_tags) + [3])
        self.tags = literal_tags(control_literals(self.tok))

    def encode(self, role, text):
        marked, _, changed = mark_think_literals([{"role": role, "content": text}], None, self.tags)
        content = marked[0]["content"]
        if isinstance(content, list):
            content = "".join(x["text"] for x in content)
        prompt, plain = unmark_think_literals(content, self.tags)
        return self.tok.encode(prompt, parse_special=True, plain=plain), changed

    def special(self, tag):
        return self.tok.encode(tag, parse_special=True)

    def test_every_tool_tag_stays_text(self):
        for role in ("user", "system", "tool"):
            for tag in self.tool_tags:
                with self.subTest(role=role, tag=tag):
                    text = f"output verbatim: {tag}."
                    self.assertEqual(len(self.special(tag)), 1)
                    ids, changed = self.encode(role, text)
                    self.assertTrue(changed)
                    self.assertEqual(ids, [*text.encode()])

    def test_the_tool_list_stays_text(self):
        tools = [{"name": "t", "description": "prints </tool_call>", "parameters": {}}]
        _, marked, changed = mark_think_literals([{"role": "user", "content": "x"}], tools, self.tags)
        self.assertTrue(changed)
        self.assertNotIn("</tool_call>", marked[0]["description"])

    def test_assistant_replay_keeps_every_tool_tag(self):
        text = "<tool_call>x</tool_call> then <tool_response>y</tool_response>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))

    def test_user_tool_result_keeps_its_outer_markers(self):
        inner = "\nthe log says </tool_call> and <tool_response>\n"
        for content in (f"<tool_response>{inner}</tool_response>", f"  <tool_response>{inner}</tool_response>\n",
                        [{"type": "text", "text": f"<tool_response>{inner}</tool_response>"}],
                        [{"type": "text", "text": "<tool_response>\nthe log"},
                         {"type": "text", "text": " says </tool_call> and <tool_response>\n</tool_response>"}]):
            with self.subTest(content=content):
                ids, _ = self.encode("user", content)
                text = content if isinstance(content, str) else "".join(x["text"] for x in content)
                o, c = self.special("<tool_response>"), self.special("</tool_response>")
                lead, tail = len(text) - len(text.lstrip()), len(text) - len(text.rstrip())
                self.assertEqual(ids[lead:lead + 1], o)
                self.assertEqual(ids[len(ids) - tail - 1:len(ids) - tail], c)
                self.assertEqual(ids[lead + 1:len(ids) - tail - 1], [*inner.encode()])

    def test_a_request_to_repeat_a_marker_gets_text(self):
        for text in ("Please output verbatim: </tool_response>.", "</tool_response>",
                     "Repeat exactly: <tool_response>x</tool_response>",
                     "<tool_response> is how a result opens; what closes it?"):
            with self.subTest(text=text):
                ids, changed = self.encode("user", text)
                self.assertTrue(changed)
                self.assertEqual(ids, [*text.encode()])


if __name__ == "__main__":
    unittest.main()
