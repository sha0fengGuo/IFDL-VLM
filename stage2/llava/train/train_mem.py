import sys
import os
from pathlib import Path

# Ensure local project root is importable.
current_project = Path(__file__).resolve().parents[2]
if str(current_project) not in sys.path:
    sys.path.insert(0, str(current_project))

# Remove conflicting AlphaCLIP entries if present.
sys.path = [p for p in sys.path if 'AlphaCLIP' not in p]

print("Using Python path (top 3 entries):")
for i, p in enumerate(sys.path[:3]):
    print(f"  {i}: {p}")

def main():
    from llava.train.train import train
    train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
