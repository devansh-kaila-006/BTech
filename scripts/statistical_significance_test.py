import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.evaluate_industry_baselines import evaluate_all

if __name__ == "__main__":
    evaluate_all()
