"""A `</think>` the reasoning quotes stays reasoning; the real end is followed by a line break."""
import unittest

from serve.frontend import OutputParser

WIDTHS = (1, 7, 100_000)


def run(text, width):
    p = OutputParser(thinking=True)
    out = []
    for i in range(0, len(text), width):
        out += p.feed(text[i:i + width])
    return out + p.finish()


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class QuotedThinkEnd(unittest.TestCase):
    def test_a_quoted_tag_stays_reasoning(self):
        for quote in ("a `</think>` could arrive here",
                      'the engine returns "</think>\\n\\n2", no call',
                      "split it on '</think>' first"):
            text = "plan\n" + quote + "\ndone\n</think>\n\nanswer"
            for width in WIDTHS:
                with self.subTest(quote=quote[:20], width=width):
                    evs = run(text, width)
                    self.assertEqual(joined(evs, "reasoning"), "plan\n" + quote + "\ndone\n")
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_the_end_after_a_quote_still_ends_the_thinking(self):
        for thought in ('she said "yes"', "it is `x`", "it is 'x'"):
            for width in WIDTHS:
                with self.subTest(thought=thought, width=width):
                    evs = run(thought + "</think>\n\nanswer", width)
                    self.assertEqual(joined(evs, "reasoning"), thought)
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_an_output_that_ends_on_the_tag(self):
        for width in WIDTHS:
            evs = run('say "</think>', width)
            self.assertEqual(joined(evs, "reasoning"), 'say "')
            self.assertEqual(joined(evs, "content"), "")

    def test_an_unquoted_tag_ends_the_thinking(self):
        for width in WIDTHS:
            evs = run("plan</think>answer", width)
            self.assertEqual(joined(evs, "reasoning"), "plan")
            self.assertEqual(joined(evs, "content"), "answer")


if __name__ == "__main__":
    unittest.main()
