import numpy as np
from utils import *

embed = get_embeddings()
mystery = embed["morning"] - embed["day"] + embed["night"]
print(get_closest_word(mystery, embed))
