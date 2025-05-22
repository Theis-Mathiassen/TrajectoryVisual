import itertools

knn_methods = [1]  
range_flags = [1, 2, 3, 4]  
similarity_systems = ["c", "a", "c+f"]  

combinations = list(itertools.product(knn_methods, range_flags, similarity_systems))
i = 0
for knn, range_flag, similarity in combinations:
    command = f"docker run -dit -e JOB_OUTPUT_DIR=/results/{i} --rm --mount type=bind,src=./results/,dst=/results/ trajectory --knn {knn} --range {range_flag} --similarity {similarity}"
    print(command) 
    i=i+1

    
