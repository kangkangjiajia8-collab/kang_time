import os
import shutil

root_dir = "outputs"

for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)

    if os.path.isdir(subdir_path):
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)

            if os.path.isfile(item_path):
                os.remove(item_path)
            else:
                shutil.rmtree(item_path)