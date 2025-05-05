import itertools

knn_methods = [1]  
range_flags = [1, 2, 3, 4]  
similarity_systems = ["c", "a", "c+f", "m"]  

combinations = list(itertools.product(knn_methods, range_flags, similarity_systems))

for knn, range_flag, similarity in combinations:
    command = f"uv run main.py --knn {knn} --range {range_flag} --similarity {similarity}"
    print(command) 

    