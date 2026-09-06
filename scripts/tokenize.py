#!/usr/bin/env python3


import argparse
import sys
from pathlib import Path

from sparklm.tokenizers.basic_tokenizer import BasicTokenizer

def main(argv=None):
    # pattern = r"(\s)" #simplest pattern to split on whitespace
    #pattern = r"([,.]|\s)" # include whitespace and punctuation as tokens
    pattern = r"([,.:;?_!\"()\']|--|\s)" # include whitespace and punctuation as tokens, but collapse consecutive whitespace into a single token
  
    with open(str(Path(__file__).parent / "the-verdict.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    
    tokenizer = BasicTokenizer(pattern)
    try:
        tokens = tokenizer.tokenize(text)
    except Exception as exc:
        print(f"Tokenization error: {exc}", file=sys.stderr)
        return 3

    out = "\n".join(tokens)
    tokenCount = len(tokens)    

    try:
        # sys.stdout.write(out)
        sys.stdout.write(f"\n\nTotal Tokens: {tokenCount}\n")
    except Exception as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 4

    print(f"Tokens: {len(tokens)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
