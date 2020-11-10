import numpy as np

def print_lines_in_file(file_name):
    num_lines = sum(1 for line in open(file_name))
    print(f"Number of lines: {num_lines}")

def get_embeddings():
    dict = {}

    file = open("data/embedding.txt", "rt")
    text = file.read()
    file.close()
    
    for entry in text.split("\n"):
        try:
            (word, array) = entry.split(" => ")
            array = list(map(float, array.split(", ")[1:-1]))
            array = np.asarray(array)

            dict[word] = array
        except:
            pass
    return dict

def get_closest_word(vec, embeddings):
    closest = ""
    min_distance = 1e100
    for (key, value) in embeddings.items():
        distance = np.linalg.norm(value - vec)
        if distance < min_distance:
            min_distance = distance
            closest = key
    return closest
