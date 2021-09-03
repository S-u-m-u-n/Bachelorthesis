import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# ToDo: Use seaborn to plot & calculate CI

parser = ArgumentParser()
parser.add_argument("--test", type=int, dest='test', nargs="?", default=1)
args = parser.parse_args()
