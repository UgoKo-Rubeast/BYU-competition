import os
import glob
import multiprocessing as mp

import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy
from PIL import Image

class ByuProcessor():
    def __init__(
        self, 
        in_dir: str = "./data/raw/train/", 
        out_dir: str = "./data/processed/", 
    ):
        super().__init__()
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.img_size = (128, 704, 704)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def process_helper(self, row):        
        # Load fpaths
        in_path= os.path.join(row["in_dir"], row["tomo_id"])
        fpaths= sorted(glob.glob(in_path + "/*"))
        z_shape= len(fpaths)

        # Load slices
        arr = [np.array(Image.open(_).convert("L")) for _ in fpaths]
        y_shape, x_shape= arr[0].shape
        arr = np.stack(arr)

        # Resize
        t,h,w= self.img_size
        zoom_factor= (t / arr.shape[0], h / arr.shape[1], w / arr.shape[2])
        arr= scipy.ndimage.zoom(arr, zoom_factor, order=1)
        arr= np.clip(arr, 0, 255).astype(np.uint8)

        # Save
        outpath= os.path.join(
            row["out_dir"], 
            "fold_{}".format(row["fold"]), 
            "{}.npy".format(row["tomo_id"]),
            )
        np.save(outpath, arr)

        return {
            "z_shape": z_shape,
            "y_shape": y_shape,
            "x_shape": x_shape,
            "fold": row["fold"],
            "tomo_id": row["tomo_id"],
        }
        

    def run(self, df):

        # Create outdir
        for _, row in df.iterrows():
            row= row.to_dict()
            fpath= os.path.join(self.out_dir, "fold_{}".format(row["fold"]))
            if not os.path.exists(fpath):
                os.mkdir(fpath)

        # Load metadata
        mp_arr= []
        for _, row in df.iterrows():
            row= row.to_dict()
            row["in_dir"]= self.in_dir
            row["out_dir"]= self.out_dir
            mp_arr.append(row)
        print("mp_arr_len:", len(mp_arr))

        # Cores
        cores= mp.cpu_count()
        print(f"Running on {cores} cores.")
        
        # Run multiprocessing
        with mp.Pool(cores) as pool:
            results= []
            for r in tqdm(pool.imap_unordered(self.process_helper, mp_arr), total=len(mp_arr)):
                results.append(r)
            results= pd.DataFrame(results)

        pool.close()
        pool.join()
        return results

if __name__ == "__main__":

    # Load folds
    df= pd.read_csv("./data/processed/folds_all.csv")
    df= df.loc[df["tomo_id"].str.startswith("tomo_"), ["tomo_id", "fold"]]
    df= df.reset_index(drop=True)
    print(df.head())

    # Process
    p= ByuProcessor(
        in_dir = "./data/raw/train/", 
        out_dir = "./data/processed/",
        )
    p.run(df=df.copy())

