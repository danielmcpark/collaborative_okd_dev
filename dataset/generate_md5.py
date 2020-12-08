import os
import hashlib

root = '/disk/cars196/'
filename = 'cars_train.tgz'

def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

fpath = os.path.join(root, filename)
md5 = calculate_md5(fpath)
print(md5)
