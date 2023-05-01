import os

repo_path = "C:/Users/Tyan/DPBCT/DeepBachTyan" # replace with your repository path
large_files = []

for root, dirs, files in os.walk(repo_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024: # files larger than 100 MB
            large_files.append(file_path)

print(large_files)
