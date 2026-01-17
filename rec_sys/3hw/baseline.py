import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import gc
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

def main(input_path_interactions: str,
         input_path_users: str,
         input_path_items: str,
         input_path_embeddings: str,
         output_path: str):
    # Загрузка данных
    print("Load data: started")
    train_df = pd.read_parquet(input_path_interactions)
    user_metadata = pd.read_parquet(input_path_users)
    item_metadata = pd.read_parquet(input_path_items)
    item_embeddings = pd.read_parquet(input_path_embeddings)
    print("Load data: finished")


    print("Preprocess data: started")
    item_metadata = item_metadata.merge(item_embeddings, on=["item_id"], how="left")
    train_df["rating"] = train_df["timespent"] > 5
    print("Preprocess data: finished")

    # als
    print("Als, preparing data (cutting data): started")
    train_df_collab_data = train_df.loc[train_df['week'] < 5].copy(deep=True)
    train_df_collab_data['rating'] = train_df_collab_data['rating'].astype(int)
    print("Als, preparing data (cutting data): started")

    print("Als, making id maps: started")
    user_id_to_uid_map = {user_id: uid for uid, user_id in enumerate(train_df_collab_data['user_id'].unique())}
    item_id_to_iid_map = {item_id: iid for iid, item_id in enumerate(train_df_collab_data['item_id'].unique())}

    train_df_collab_data['uid'] = train_df_collab_data['user_id'].map(user_id_to_uid_map)
    train_df_collab_data['iid'] = train_df_collab_data['item_id'].map(item_id_to_iid_map)
    print("Als, making id maps: finished")

    print("Als, making csr_matrix: started")
    user_item_matrix = csr_matrix(
        (train_df_collab_data['rating'], (train_df_collab_data['uid'], train_df_collab_data['iid'])))
    print("Als, making csr_matrix: finished")

    print("Als, training: started")
    als = AlternatingLeastSquares(random_state=42)

    als.fit(user_item_matrix)
    print("Als, training: finished")

    print("Als, scoring on remaining train: started")
    remaining_train_df = train_df.loc[train_df['week'] >= 5].copy(deep=True)


    train_users = pd.DataFrame({"user_id": remaining_train_df["user_id"].unique()})
    train_items = pd.DataFrame({"item_id": remaining_train_df["item_id"].unique()})

    remaining_train_df['uid'] = remaining_train_df['user_id'].map(user_id_to_uid_map)
    remaining_train_df['iid'] = remaining_train_df['item_id'].map(item_id_to_iid_map)

    remaining_train_df['als_score'] = remaining_train_df.apply(
        lambda r: als.recommend(int(r.uid), user_item_matrix, filter_already_liked_items=False, items=[int(r.iid)])[1][
            0]
        if ~np.isnan(r.uid) and ~np.isnan(r.iid) else np.nan, axis=1)
    print("Als, scoring on remaining train: finished")

    print("Preparing train dataset for ranking: started")
    features = ["age", "gender", "author_id", "duration", "als_score"]



    train = (remaining_train_df
             .merge(user_metadata, on=["user_id"], how="inner")
             .merge(item_metadata, on=["item_id"], how="inner")
             )[["user_id", "item_id", "age", "gender", "author_id", "duration", "als_score", "rating"]]


    _, group_train = np.unique(train['user_id'], return_counts=True)

    train.sort_values("user_id", inplace=True)
    print("Preparing train dataset for ranking: finished")

    print("Training ranker: started")
    lgb_train = lgb.Dataset(
        train[features],
        train["rating"],
        categorical_feature=["age", "gender", "author_id"],
        group=group_train
    )

    params = {
        'objective': 'lambdarank',  # For binary classification
        'metric': 'auc',  # Evaluation metric
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'num_leaves': 31,  # Max number of leaves in one tree
        'learning_rate': 0.05,  # Step size shrinkage
        'feature_fraction': 0.9,  # Fraction of features considered at each split
        'verbose': -1  # Suppress verbose output during training
    }


    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,  # Number of boosting rounds (iterations)
                    )
    print("Training ranker: finished")

    print("Ranker, scoring on remaining train: started")

    sample_items = train_items.sample(frac=0.05, random_state=314)[["item_id"]]
    prediction_df = (pd
                     .merge(train_users, sample_items, how="cross")
                     .merge(user_metadata[["user_id", "age", "gender"]], on=["user_id"], how="inner")
                     .merge(item_metadata[["item_id", "author_id", "duration"]], on=["item_id"], how="inner")
                     .merge(remaining_train_df[["user_id", "item_id", "rating", "als_score"]], on=["user_id", "item_id"], how="left")
                     .query("rating.isnull()", engine="python")
                     )


    y_pred = gbm.predict(prediction_df[features])

    prediction_df["prediction"] = y_pred

    prediction_df["item_score"] = prediction_df.apply(lambda r: (r.item_id, r.prediction), axis=1)

    prediction_by_user = (prediction_df
                          .groupby("user_id", as_index=False)
                          .agg(scores_list=("item_score", list))
                          .reset_index()
                          )


    del prediction_df
    gc.collect()

    prediction_by_user["recs"] = prediction_by_user.apply(
        lambda r: [i[0] for i in sorted(r.scores_list, key=lambda x: x[1], reverse=True)[:10]], axis=1)

    result = prediction_by_user.explode("recs")
    print("Ranker, scoring on remaining train: finished")

    print(f"Saving result to file {output_path}.")
    result.to_csv(output_path, index=False)
    print("File saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path_interactions", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--input_path_users", type=str, required=True, help="Input path to users demographic info parquet file")
    parser.add_argument("--input_path_items", type=str, required=True, help="Input path to items attributive info parquet file")
    parser.add_argument("--input_path_embeddings", type=str, required=True, help="Input path to items embeddings parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path_interactions,
         args.input_path_users,
         args.input_path_items,
         args.input_path_embeddings,
         args.output_path)