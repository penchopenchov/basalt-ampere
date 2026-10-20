#!/usr/bin/env python3
"""Closing the engine's stdin mid-generation stops the running request and ends the process cleanly.

    stdin_stop_test.py ENGINE MODEL.basalt IDS_FILE [ENGINE ARGS...]

Starts `ENGINE --serve`, asks for a long greedy generation, closes stdin after a few tokens, and checks that the
request stops within a few windows and the engine exits with status 0 soon after.  Needs a GPU and a model pack.
"""
import subprocess
import sys
import time

MAX_NEW = 3000
EXIT_WITHIN_S = 30.0


def main() -> int:
    engine, model, ids_file, *extra = sys.argv[1:]
    ids = open(ids_file).readline().strip()
    proc = subprocess.Popen([engine, "--serve", "--model", model, *extra], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            if line.startswith("READY"):
                break
        else:
            print("stdin_stop_test: the engine never became ready")
            return 1
        proc.stdin.write(f"GEN {MAX_NEW} temperature=0 {ids}\n")
        proc.stdin.flush()
        tokens = 0
        for line in proc.stdout:
            if line.startswith("T "):
                tokens += 1
                if tokens == 8:
                    break
        else:
            print("stdin_stop_test: the request ended before stdin was closed")
            return 1
        proc.stdin.close()
        closed_at = time.monotonic()
        for line in proc.stdout:
            if line.startswith("T "):
                tokens += 1
        status = proc.wait(timeout=EXIT_WITHIN_S)
        took = time.monotonic() - closed_at
    except subprocess.TimeoutExpired:
        print(f"stdin_stop_test: the engine was still running {EXIT_WITHIN_S:.0f} s after stdin closed")
        return 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    print(f"stdin_stop_test: {tokens} of {MAX_NEW} tokens, exit status {status}, {took:.2f} s after the close")
    if tokens >= 200:
        print("stdin_stop_test: the request ran on after stdin closed")
        return 1
    if status != 0:
        print("stdin_stop_test: the engine did not exit cleanly")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
