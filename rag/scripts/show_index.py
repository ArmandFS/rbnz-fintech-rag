import sys
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path(__file__).resolve().parents[1]))

from db import get_index_summary


def main() -> None:
    pprint(get_index_summary())


if __name__ == "__main__":
    main()
