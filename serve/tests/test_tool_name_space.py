"""A tool call written `<function= NAME>` is the call NAME."""
import unittest

from serve.frontend import OutputParser, parse_tool_call

SCHEMA = [{"name": "capture", "parameters": {"properties": {"path": {"type": "string"}}}}]
TEXT = "ok</think>\n\n<tool_call>\n<function= capture>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>"


def events(width, stream_tools):
    p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=stream_tools)
    out = []
    for i in range(0, len(TEXT), width):
        out += p.feed(TEXT[i:i + width])
    return out + p.finish()


class NameSpace(unittest.TestCase):
    def test_parse_tool_call(self):
        self.assertEqual(parse_tool_call("<function= capture>\n</function>").name, "capture")

    def test_whole_and_streamed(self):
        for stream_tools in (False, True):
            for width in (3, 100_000):
                with self.subTest(stream_tools=stream_tools, width=width):
                    evs = events(width, stream_tools)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([c.name for c in calls], ["capture"])
                    self.assertEqual(calls[0].arguments, {"path": "x"})
                    self.assertEqual({e.call.name for e in evs if e.kind == "tool_start"} - {"capture"}, set())


if __name__ == "__main__":
    unittest.main()
