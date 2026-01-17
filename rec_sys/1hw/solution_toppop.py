import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import argparse

def main(input_path: str, output_path: str):
    train = pd.read_parquet(input_path)
    
    train_als = train.copy()
    
    train_als['confidence'] = 1.0
    
    train_als.loc[train_als['like'], 'confidence'] += 4.0
    train_als.loc[train_als['share'], 'confidence'] += 3.0
    train_als.loc[train_als['bookmark'], 'confidence'] += 3.0
    train_als.loc[train_als['click_on_author'], 'confidence'] += 2.0
    train_als.loc[train_als['open_comments'], 'confidence'] += 1.0
    
    train_als['time_weight'] = train_als['timespent'] / 255.0
    train_als['confidence'] += train_als['time_weight'] * 2.0
    
    train_als.loc[train_als['dislike'], 'confidence'] = 0.1
    
    train_als['user_id_cat'] = train_als['user_id'].astype('category')
    train_als['item_id_cat'] = train_als['item_id'].astype('category')

    user_cat_map = dict(enumerate(train_als['user_id_cat'].cat.categories))
    item_cat_map = dict(enumerate(train_als['item_id_cat'].cat.categories))
    
    user_id_to_cat = {v: k for k, v in user_cat_map.items()}
    item_id_to_cat = {v: k for k, v in item_cat_map.items()}
    
    user_item_matrix = csr_matrix((
        train_als['confidence'].astype(float),
        (train_als['user_id_cat'].cat.codes, train_als['item_id_cat'].cat.codes)
    ))

    model = implicit.als.AlternatingLeastSquares(
        factors=128,
        regularization=0.05,
        iterations=30,
        random_state=42,
        use_gpu=False,
        alpha=15.0
    )

    model.fit(user_item_matrix)

    all_original_user_ids = train['user_id'].unique()
    
    users_in_als_model = set(train_als['user_id'].unique())
    
    users_to_recommend_als = []
    users_fallback_pop = []
    
    for uid in all_original_user_ids:
        if uid in users_in_als_model:
            users_to_recommend_als.append(uid)
        else:
            users_fallback_pop.append(uid)

    results_list = []
    N = 10

    if users_to_recommend_als:
        user_codes_to_rec = [user_id_to_cat[uid] for uid in users_to_recommend_als]
        
        als_recs_codes, scores = model.recommend(
            userid=user_codes_to_rec,
            user_items=user_item_matrix[user_codes_to_rec],
            N=N * 2,
            filter_already_liked_items=True,
            recalculate_user=True
        )
        
        for i, user_code in enumerate(user_codes_to_rec):
            original_user_id = user_cat_map[user_code]
            
            user_interactions = train_als[train_als['user_id'] == original_user_id]
            watched_items = set(user_interactions['item_id'])
            
            recommended_items = []
            for item_code in als_recs_codes[i]:
                item_id = item_cat_map[item_code]
                if item_id not in watched_items:
                    recommended_items.append(item_id)
                if len(recommended_items) >= N:
                    break
            
            if len(recommended_items) < N:
                additional_needed = N - len(recommended_items)
                popular_items = get_popular_items(train, exclude_items=watched_items, n=additional_needed)
                recommended_items.extend(popular_items)
            
            for item_id in recommended_items[:N]:
                results_list.append((original_user_id, item_id))

    if users_fallback_pop:
        top_popular_items = get_popular_items(train, n=N)
        
        for user_id in users_fallback_pop:
            for item_id in top_popular_items:
                results_list.append((user_id, item_id))

    final_df = pd.DataFrame(results_list, columns=['user_id', 'recs'])
    
    final_df['user_id'] = final_df['user_id'].astype(np.uint32)
    final_df['recs'] = final_df['recs'].astype(np.uint32)
    
    final_df.to_csv(output_path, index=False)

def get_popular_items(train, exclude_items=None, n=10):
    item_popularity = train.groupby('item_id').agg({
        'user_id': 'count',
        'like': 'sum',
        'timespent': 'mean'
    }).reset_index()
    
    item_popularity['popularity_score'] = (
        item_popularity['user_id'] * 0.5 +
        item_popularity['like'] * 2.0 +
        item_popularity['timespent'] * 0.1
    )
    
    if exclude_items:
        item_popularity = item_popularity[~item_popularity['item_id'].isin(exclude_items)]
    
    top_items = item_popularity.nlargest(n, 'popularity_score')['item_id'].tolist()
    
    if len(top_items) < n:
        all_items = item_popularity.nlargest(n * 2, 'popularity_score')['item_id'].tolist()
        additional_items = [item for item in all_items if item not in top_items][:n - len(top_items)]
        top_items.extend(additional_items)
    
    return top_items[:n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path, args.output_path)