# sphere_appro/__main__.py
import logging
from .config import parse_args
from .event import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def main():
    config = parse_args()
    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == '__main__':
    main()
