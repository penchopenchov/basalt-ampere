#!/usr/bin/env python3
"""The engine enforces a request's grammar on the tokens it samples.

    grammar_engine_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

Starts `ENGINE --serve --grammar on` and asks for a person as JSON under a schema that only allows the name "Bob":
the model's own answer names someone else, so a sampled token violates the grammar.  With the grammar key the reply
must parse, carry the name "Bob" and the engine must report a masked resample; the same request without the key
must not (the control).   Needs a GPU and a model pack.
"""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import model_file  # noqa: E402
import tokenizer as ST  # noqa: E402

PROMPT = ("<|im_start|>user\nReply with only a JSON object that has the keys \"name\" and \"age\".<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")
MAX_NEW = 80


def load_tokenizer(model: str):
    mf = model_file.ModelFile(model)
    vocab = json.loads(mf.read("tokenizer/vocab.json"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = mf.read("tokenizer/merges.txt").decode().split("\n")
    types = json.loads(mf.read("tokenizer/token_type.json"))
    return ST.Tokenizer(tokens, merges, types), vocab


def text_of(tok, vocab, ids):
    ends = {vocab["<|im_end|>"], vocab["<|endoftext|>"]}
    return tok.decode([i for i in ids if i not in ends]).strip()


def request(proc, ids, extra: str):
    proc.stdin.write(f"GEN {MAX_NEW} temperature=0 {extra} {','.join(map(str, ids))}\n")
    proc.stdin.flush()
    got = []
    for line in proc.stdout:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return got
        elif line.startswith("ERR"):
            raise RuntimeError(line.strip())
    raise RuntimeError("the engine ended")


def main() -> int:
    engine, model, *extra = sys.argv[1:]
    tok, vocab = load_tokenizer(model)
    ids = tok.encode(PROMPT, parse_special=True)
    schema = {"type": "object", "properties": {"name": {"enum": ["Bob"]}, "age": {"type": "integer"}},
              "required": ["name", "age"], "additionalProperties": False}
    grammar = "grammar=" + json.dumps({"kind": "json_schema", "schema": schema}, separators=(",", ":")).replace(" ", "\\u0020")
    with tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen([engine, "--serve", "--model", model, "--grammar", "on", *extra], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                if line.startswith("READY"):
                    break
            else:
                print("grammar_engine_test: the engine never became ready")
                return 1
            control = text_of(tok, vocab, request(proc, ids, ""))
            text = text_of(tok, vocab, request(proc, ids, grammar))
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        err.seek(0)
        log = err.read()
    resamples = [int(m) for m in re.findall(r"(\d+) masked resamples", log)]
    print(f"control : {control!r}\nenforced: {text!r}\nmasked resamples per request: {resamples}")
    ok = True
    try:
        doc = json.loads(text)
        if doc.get("name") != "Bob" or not isinstance(doc.get("age"), int):
            raise ValueError("wrong content")
    except (ValueError, AttributeError) as e:
        print(f"FAIL: the grammar reply does not satisfy the schema: {e}")
        ok = False
    if len(resamples) != 1 or resamples[0] < 1:
        print("FAIL: the request should show at least one masked resample")
        ok = False
    if '"Bob"' in control:
        print("FAIL: the control already satisfies the grammar, so the test proves nothing")
        ok = False
    print("grammar_engine_test:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
