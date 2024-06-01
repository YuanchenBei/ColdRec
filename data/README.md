## 🌿 Included Datasets

Currently, we include **MovieLens**, **Book-Crossing**, **CiteULike**, and **XING**. We will continue to update appropriate and new datasets.

The statistics of the included datasets are as follows:
|  **Dataset** | **Users** | **Items** | **Interactions** | **User Contents** | **Item Contents** | 
|:---------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|  MovieLens   | 6,040 |  3,706 | 1,000,209 | 3,469 | 206 |
|  Book-Crossing | 92,107 | 270,170 | 1,031,175 | 2,312 | 416 |
|  CiteULike  |  5,551 | 16,980 | 204,986 | N/A | 300 |
|  XING   |  106,881 | 20,519 | 3,856,580 | N/A | 2,738 |

The link (Google Drive) to access the datasets is available at: https://drive.google.com/drive/folders/13JJ25vf5dpFzxe1ITQYONrEIUAsB2ZU8?usp=sharing.

## 🌴 Dataset Processing
We have provided the code to directly dataset processing. You can process the dataset into training format in the following two steps:

``` bash
python split.py --dataset [DATASET NAME] --cold_object [user/item]

python convert.py --dataset [DATASET NAME] --cold_object [user/item]
```

In the above scripts, the **[DATASET NAME]** for --dataset should be replaced by your target dataset name, such as movielens. Then, the **[user/item]** for --cold_object should be selected as user or item for the user cold-start setting or item cold-start setting.