from time import sleep

from scrap import batch_download_parallel_safe, LOG_DIR
from hash import embed_existing_galleries
# from cluster_sep import cluster_all_batches
# from merge import merge_cluster_batches
import gc
import math
import os, psutil
import resource


def delete_folder(folder_path,no_print=False):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        if not no_print:
            print(f"✅ Deleted folder: {folder_path}")
    else:
        if not no_print:
            print(f"⚠️ Folder not found: {folder_path}")

import os
import shutil

def rename_merged_to_clusters(merged_dir="merged_clusters", target_dir="clusters"):
    # Rename "merged_clusters" to "clusters"
    if os.path.exists(merged_dir):
        if os.path.lexists(target_dir):
            raise FileExistsError(f"Refusing to replace existing clusters: {target_dir}")
        os.rename(merged_dir, target_dir)
        print(f"✅ Renamed '{merged_dir}' → '{target_dir}'")
    else:
        print(f"❌ '{merged_dir}' does not exist.")

#   def main_no_download():
#       embed_existing_galleries()
#
#       cluster_all_batches()
#
#       merge_cluster_batches()
#
#       delete_folder('vectors')
#       delete_folder('clusters')
#       rename_merged_to_clusters()

def count_webp_images(folder_path='galleries'):
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.webp'):
                image_count += 1
    return image_count

import orjson
def delete_bad(folder_path='galleries', meta_file = 'metadata.json'):
    if not os.path.isdir(folder_path):
        return
    with open(meta_file, "rb") as f:
        existing_metadata = orjson.loads(f.read())

    if not isinstance(existing_metadata, list) or any(
        not isinstance(entry, dict) or type(entry.get('id')) is not int or entry['id'] <= 0
        for entry in existing_metadata
    ):
        raise ValueError(f"Invalid gallery IDs in metadata: {meta_file}")

    I = frozenset([m['id'] for m in existing_metadata])

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if name.isdigit() and os.path.isdir(path) and int(name) not in I:
            delete_folder(path,no_print=True)

if __name__ == '__main__':
    max_id = 697_150
    offset = 697_100
    download_batch = 10
    hash_cut_off = pow(2,13) # 16384
    cluster_offset = 0
    cur_batch = 0
    total_downloaded = 0

    os.makedirs(LOG_DIR, exist_ok=True)
    # delete_bad()
    print('Total number of batches',math.ceil((max_id-offset+1)/download_batch))
    while offset+cur_batch*download_batch<=max_id:
        downloaded = 0
        last_flush = 0
        while downloaded<=hash_cut_off:
            last_flush+=1
            if last_flush>10:
                last_flush = 0
                gc.collect()

            batch_end = min(offset+(cur_batch+1)*download_batch, max_id+1)
            previous_downloaded = count_webp_images()
            batch_download_parallel_safe(offset+cur_batch*download_batch, batch_end,batch_id=cur_batch)
            gc.collect()

            cur_batch+=1
            downloaded = count_webp_images()
            total_downloaded += max(downloaded - previous_downloaded, 0)
            print('Downloaded',downloaded,'images')

            print('Galleries till', batch_end-1)

            p = psutil.Process(os.getpid())
            if  p.memory_info().rss / 1e9>8: # might as well restart at this point
                raise Exception('Memory usage exceeds 8GB')

            print("RSS:", p.memory_info().rss / 1e9, "GB")
            # print("Peak RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "GB")

            with open(os.path.join(LOG_DIR, "download_count.txt"), "a") as f:
                f.write(f"{batch_end-1}\n")
                f.flush()
                f.close()

            if offset+cur_batch*download_batch>max_id:
                print('Max_ID =',max_id,'reached terminating downloads.')
                break
            if downloaded<hash_cut_off:
                print('Downloading another batch since cut off is',hash_cut_off,'images.')
            else:
                print('Starting hashing due to exceeding cut off of',hash_cut_off,'images.')
        # Sometimes the deletion handling breaks, i.e., bad folders stay, this ensures no "half-downloaded" files
        delete_bad()

        if os.path.isdir('galleries'):
            embed_existing_galleries()
            if any(name.isdigit() and os.path.isdir(os.path.join('galleries', name))
                   for name in os.listdir('galleries')):
                raise RuntimeError('Some galleries could not be fully embedded; source images were preserved.')
        # delete_folder('logging')
        gc.collect()


    print('Total images downloaded:',total_downloaded)
