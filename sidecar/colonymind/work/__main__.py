"""Explicit offline repair: python -m colonymind.work (configured state directory)."""
from . import repair_work

if __name__ == '__main__':
    print(f'Rebuilt {repair_work()} work roots')
