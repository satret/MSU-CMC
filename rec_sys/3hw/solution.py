import pandas as pd
import numpy as np
import lightgbm as lgb
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
import re
import gc
import os

def main(input_path_interactions: str = None,
         input_path_users: str = None,
         input_path_items: str = None,
         input_path_embeddings: str = None,
         output_path: str = None):
    
    if input_path_interactions is None:
        input_path_interactions = "train.parquet"
    if input_path_users is None:
        input_path_users = "user_metadata.parquet"
    if input_path_items is None:
        input_path_items = "item_metadata.parquet"
    if input_path_embeddings is None:
        input_path_embeddings = "item_embeddings.parquet"
    if output_path is None:
        output_path = "submission.csv"
    
    for path in [input_path_interactions, input_path_users, input_path_items, input_path_embeddings]:
        if not os.path.exists(path):
            pd.DataFrame(columns=['user_id', 'recs']).to_csv(output_path, index=False)
            return
    
    try:
        train = pd.read_parquet(input_path_interactions)
        user_meta = pd.read_parquet(input_path_users)
        item_meta = pd.read_parquet(input_path_items)
        item_emb = pd.read_parquet(input_path_embeddings)
     
        train['timespent_norm'] = np.clip(train['timespent'] / 60.0, 0, 1)
        
        train['score'] = (
            train['timespent_norm'] * 1.0 +
            train['like'] * 3.0 +
            train['share'] * 4.0 +
            train['bookmark'] * 4.0 +
            train['click_on_author'] * 2.0 +
            train['open_comments'] * 1.5 -
            train['dislike'] * 5.0
        )
        
        train['target_binary'] = (
            (train['timespent'] > 15) |
            (train['like'] == 1) | 
            (train['share'] == 1) |
            (train['bookmark'] == 1)
        ).astype(int)
        
        emb_matrix = np.stack(item_emb['embedding'].values)
        emb_columns = [f'emb_{i}' for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, columns=emb_columns)
        emb_df['item_id'] = item_emb['item_id'].values
        
        unique_users = train['user_id'].unique()
        unique_items = train['item_id'].unique()
        
        user_map = {u: i for i, u in enumerate(unique_users)}
        item_map = {u: i for i, u in enumerate(unique_items)}
        inv_item_map = {i: u for i, u in enumerate(unique_items)}
        
        train_als = train.copy()
        train_als['uid'] = train_als['user_id'].map(user_map)
        train_als['iid'] = train_als['item_id'].map(item_map)
        train_als = train_als.dropna(subset=['uid', 'iid'])
        
        if not train_als.empty:
            train_als['uid'] = train_als['uid'].astype(int)
            train_als['iid'] = train_als['iid'].astype(int)
            sparse_matrix = sp.csr_matrix(
                (train_als['score'].astype(np.float32), 
                 (train_als['uid'], train_als['iid'])),
                shape=(len(unique_users), len(unique_items))
            )
            als_model = AlternatingLeastSquares(
                factors=128,
                regularization=0.01,
                iterations=20,
                random_state=42,
                use_gpu=False
            )
            als_model.fit(sparse_matrix)
            candidates_als = []
            
            for user_id in unique_users[:5000]:
                if user_id in user_map:
                    uid = user_map[user_id]
                    try:
                        ids, scores = als_model.recommend(
                            uid, 
                            sparse_matrix[uid], 
                            N=50,
                            filter_already_liked_items=True
                        )
                        
                        for item_idx, score in zip(ids, scores):
                            if item_idx in inv_item_map:
                                candidates_als.append({
                                    'user_id': user_id,
                                    'item_id': inv_item_map[item_idx],
                                    'als_score': float(score)
                                })
                    except:
                        continue
            
            candidates_df = pd.DataFrame(candidates_als)
        else:
            candidates_df = pd.DataFrame()
       
        item_popularity = train.groupby('item_id')['score'].agg(['sum', 'count']).reset_index()
        item_popularity.columns = ['item_id', 'total_score', 'interaction_count']
        
        item_popularity['popularity_score'] = (
            item_popularity['total_score'] / item_popularity['total_score'].max() * 0.7 +
            item_popularity['interaction_count'] / item_popularity['interaction_count'].max() * 0.3
        )
        top_popular = item_popularity.nlargest(200, 'popularity_score')['item_id'].tolist()
        user_viewed = train.groupby('user_id')['item_id'].apply(set).to_dict()
        
        popular_candidates = []
            viewed = user_viewed.get(user_id, set())
            for item_id in top_popular:
                if item_id not in viewed:
                    popular_candidates.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'popularity_score': item_popularity.loc[item_popularity['item_id'] == item_id, 'popularity_score'].values[0]
                    })
                if len([c for c in popular_candidates if c['user_id'] == user_id]) >= 30:
                    break
        
        popular_df = pd.DataFrame(popular_candidates)
        
        if not candidates_df.empty and not popular_df.empty:
            all_candidates = pd.concat([candidates_df, popular_df], ignore_index=True)
            all_candidates = all_candidates.drop_duplicates(subset=['user_id', 'item_id'])
        elif not candidates_df.empty:
            all_candidates = candidates_df
        else:
            all_candidates = popular_df

        all_candidates = all_candidates.merge(user_meta, on='user_id', how='left')

        item_meta_clean = item_meta.drop(columns=['embedding'], errors='ignore')
        all_candidates = all_candidates.merge(item_meta_clean, on='item_id', how='left')

        all_candidates = all_candidates.merge(emb_df, on='item_id', how='left')

        user_stats = train.groupby('user_id').agg({
            'score': 'sum',
            'timespent': 'mean',
            'like': 'sum'
        }).reset_index()
        user_stats.columns = ['user_id', 'user_total_score', 'user_avg_timespent', 'user_like_count']
        
        item_stats = train.groupby('item_id').agg({
            'score': 'sum',
            'user_id': 'count',
            'like': 'mean'
        }).reset_index()
        item_stats.columns = ['item_id', 'item_total_score', 'item_view_count', 'item_like_rate']
        
        all_candidates = all_candidates.merge(user_stats, on='user_id', how='left')
        all_candidates = all_candidates.merge(item_stats, on='item_id', how='left')
        
        numeric_cols = all_candidates.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if all_candidates[col].isnull().any():
                if col == 'age':
                    all_candidates[col] = all_candidates[col].fillna(25)
                elif col == 'duration':
                    all_candidates[col] = all_candidates[col].fillna(0)
                else:
                    all_candidates[col] = all_candidates[col].fillna(all_candidates[col].median() if not all_candidates[col].isnull().all() else 0)
        
        train_for_lgb = train.copy()
        train_for_lgb = train_for_lgb.merge(user_meta, on='user_id', how='left')
        train_for_lgb = train_for_lgb.merge(item_meta_clean, on='item_id', how='left')
        train_for_lgb = train_for_lgb.merge(emb_df, on='item_id', how='left')
        
        positives = train_for_lgb[train_for_lgb['target_binary'] == 1]
        negatives = train_for_lgb[train_for_lgb['target_binary'] == 0]
        
        if len(positives) > 100000:
            positives = positives.sample(100000, random_state=42)
        if len(negatives) > len(positives) * 2:
            negatives = negatives.sample(len(positives) * 2, random_state=42)
        
        balanced_train = pd.concat([positives, negatives])
        
        drop_cols = ['user_id', 'item_id', 'target_binary', 'score', 
                     'timespent', 'like', 'dislike', 'share', 'bookmark', 
                     'click_on_author', 'open_comments', 'timespent_norm',
                     'platform', 'agent', 'place', 'uid', 'iid', 'als_score',
                     'popularity_score', 'embedding']
        
        feature_cols = []
        for col in balanced_train.columns:
            if col in drop_cols:
                continue
            if str(col).startswith('Unnamed'):
                continue
            if pd.api.types.is_numeric_dtype(balanced_train[col]):
                feature_cols.append(col)
        
        
        X_train = balanced_train[feature_cols].copy()
        y_train = balanced_train['target_binary']
        
        X_train = X_train.fillna(0)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'num_leaves': 63,
            'max_depth': 8,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': 1
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        gbm = lgb.train(params, lgb_train, num_boost_round=200)
        
        missing_cols = set(feature_cols) - set(all_candidates.columns)
        for col in missing_cols:
            all_candidates[col] = 0
        
        X_pred = all_candidates[feature_cols].copy()
        X_pred = X_pred.fillna(0)
        
        all_candidates['prediction'] = gbm.predict(X_pred)
        
        if 'als_score' in all_candidates.columns:
            all_candidates['als_score'] = all_candidates['als_score'].fillna(0)
            if all_candidates['als_score'].max() > all_candidates['als_score'].min():
                all_candidates['als_score_norm'] = (
                    all_candidates['als_score'] - all_candidates['als_score'].min()
                ) / (all_candidates['als_score'].max() - all_candidates['als_score'].min())
            else:
                all_candidates['als_score_norm'] = 0
        else:
            all_candidates['als_score_norm'] = 0
            
        if 'popularity_score' in all_candidates.columns:
            all_candidates['popularity_score'] = all_candidates['popularity_score'].fillna(0)
        
        all_candidates['final_score'] = (
            all_candidates['prediction'] * 0.7 +
            all_candidates['als_score_norm'] * 0.2 +
            all_candidates.get('popularity_score', 0) * 0.1
        )
        
        
        all_candidates = all_candidates.sort_values(['user_id', 'final_score'], ascending=[True, False])
        top_recs = all_candidates.groupby('user_id').head(10)
        
        result = top_recs[['user_id', 'item_id']].rename(columns={'item_id': 'recs'})
        
        if len(result) < 50000:
            for user_id in all_candidates['user_id'].unique():
                user_recs = result[result['user_id'] == user_id]
                if len(user_recs) < 10:
                    needed = 10 - len(user_recs)
                    for item_id in top_popular[:needed]:
                        result = pd.concat([
                            result, 
                            pd.DataFrame([{'user_id': user_id, 'recs': item_id}])
                        ], ignore_index=True)
        
        result.to_csv(output_path, index=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            if 'train' in locals():
                popular_items = train.groupby('item_id').size().nlargest(10).index.tolist()
                user_ids = train['user_id'].unique()[:1000]
                result = []
                for user_id in user_ids:
                    for item_id in popular_items:
                        result.append({'user_id': user_id, 'recs': item_id})
                pd.DataFrame(result).to_csv(output_path, index=False)
            else:
                pd.DataFrame(columns=['user_id', 'recs']).to_csv(output_path, index=False)
        except:
            pd.DataFrame(columns=['user_id', 'recs']).to_csv(output_path, index=False)

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Recommender arguments.")
        parser.add_argument("--input_path_interactions", type=str, required=False, default="train.parquet", help="Input path to train parquet file")
        parser.add_argument("--input_path_users", type=str, required=False, default="user_metadata.parquet", help="Input path to users demographic info parquet file")
        parser.add_argument("--input_path_items", type=str, required=False, default="item_metadata.parquet", help="Input path to items attributive info parquet file")
        parser.add_argument("--input_path_embeddings", type=str, required=False, default="item_embeddings.parquet", help="Input path to items embeddings parquet file")
        parser.add_argument("--output_path", type=str, required=False, default="submission.csv", help="Output path to csv with recommendations")
        
        args = parser.parse_args()
        main(args.input_path_interactions,
             args.input_path_users,
             args.input_path_items,
             args.input_path_embeddings,
             args.output_path)
    except SystemExit:
        main()