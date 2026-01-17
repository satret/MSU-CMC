import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, bm25_weight
import lightgbm as lgb
import argparse
import os
import gc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def create_sparse_matrix(train_df, user_col='user_id', item_col='item_id'):
    """Create sparse user-item matrix from interactions"""
    # Create label encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Fit and transform
    user_ids = user_encoder.fit_transform(train_df[user_col])
    item_ids = item_encoder.fit_transform(train_df[item_col])
    
    # Create sparse matrix
    sparse_matrix = csr_matrix(
        (np.ones(len(train_df)), (user_ids, item_ids)),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )
    
    return sparse_matrix, user_encoder, item_encoder

def main(input_path_interactions: str = None,
         input_path_users: str = None,
         input_path_items: str = None,
         input_path_embeddings: str = None,
         output_path: str = None):
    
    # Default paths
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
    
    # Check files existence
    for path, name in [
        (input_path_interactions, "interactions"),
        (input_path_users, "users"),
        (input_path_items, "items"),
        (input_path_embeddings, "embeddings")
    ]:
        if not os.path.exists(path):
            print(f"Warning: {name} file {path} not found. Creating empty submission.")
            pd.DataFrame(columns=['user_id', 'recs']).to_csv(output_path, index=False)
            return
    
    try:
        print("Loading data...")
        # Load data
        train_df = pd.read_parquet(input_path_interactions)
        user_metadata = pd.read_parquet(input_path_users)
        item_metadata = pd.read_parquet(input_path_items)
        item_embeddings = pd.read_parquet(input_path_embeddings)
        
        print(f"Loaded {len(train_df)} interactions, {len(user_metadata)} users, {len(item_metadata)} items")
        
        # Create engagement score
        print("Creating engagement features...")
        # Normalize and combine engagement signals
        for col in ['timespent', 'like', 'dislike', 'share', 'bookmark']:
            if col in train_df.columns:
                if train_df[col].dtype == bool:
                    train_df[col] = train_df[col].astype(float)
                else:
                    train_df[col] = train_df[col].astype(float) / train_df[col].max()
        
        train_df['engagement_score'] = (
            train_df.get('timespent', 0) * 0.4 +
            train_df.get('like', 0) * 0.3 +
            train_df.get('share', 0) * 0.2 +
            train_df.get('bookmark', 0) * 0.1
        )
        
        # Fill missing engagement scores with mean
        if train_df['engagement_score'].isnull().any():
            train_df['engagement_score'] = train_df['engagement_score'].fillna(
                train_df['engagement_score'].mean()
            )
        
        # Get all unique users and items
        all_users = train_df['user_id'].unique()
        all_items = train_df['item_id'].unique()
        
        print(f"Unique users: {len(all_users)}, unique items: {len(all_items)}")
        
        # 1. Popularity-based recommendations (fallback)
        print("Calculating popular items...")
        item_popularity = train_df.groupby('item_id')['engagement_score'].sum().reset_index()
        item_popularity = item_popularity.sort_values('engagement_score', ascending=False)
        popular_items = item_popularity['item_id'].head(500).tolist()
        
        # 2. User-based statistics
        print("Calculating user statistics...")
        user_stats = train_df.groupby('user_id').agg({
            'engagement_score': ['sum', 'mean', 'count'],
            'item_id': 'nunique'
        }).reset_index()
        user_stats.columns = ['user_id', 'user_eng_sum', 'user_eng_mean', 'user_interactions', 'user_unique_items']
        
        # 3. Item-based statistics
        print("Calculating item statistics...")
        item_stats = train_df.groupby('item_id').agg({
            'engagement_score': ['sum', 'mean', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        item_stats.columns = ['item_id', 'item_eng_sum', 'item_eng_mean', 'item_interactions', 'item_unique_users']
        
        # 4. Prepare ALS matrix
        print("Preparing ALS matrix...")
        # Create sparse matrix for ALS
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        train_df['user_idx'] = user_encoder.fit_transform(train_df['user_id'])
        train_df['item_idx'] = item_encoder.fit_transform(train_df['item_id'])
        
        # Create sparse matrix with engagement scores as weights
        user_item_matrix = coo_matrix(
            (train_df['engagement_score'].values,
             (train_df['user_idx'].values, train_df['item_idx'].values))
        ).tocsr()
        
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        
        # 5. Train ALS model
        print("Training ALS model...")
        try:
            als_model = AlternatingLeastSquares(
                factors=32,
                regularization=0.1,
                iterations=15,
                random_state=42,
                use_gpu=False
            )
            
            # Use bm25 weighting for better results
            weighted_matrix = bm25_weight(user_item_matrix, K1=100, B=0.8)
            
            als_model.fit(weighted_matrix)
            print("ALS model trained successfully")
            
            # Generate ALS recommendations
            als_recommendations = {}
            batch_size = 1000
            
            for i in range(0, len(all_users), batch_size):
                batch_users = all_users[i:i+batch_size]
                user_indices = user_encoder.transform(batch_users)
                
                for user_id, user_idx in zip(batch_users, user_indices):
                    try:
                        # Get recommendations
                        item_indices, scores = als_model.recommend(
                            user_idx,
                            user_item_matrix[user_idx],
                            N=50,
                            filter_already_liked_items=True
                        )
                        
                        if len(item_indices) > 0:
                            # Convert indices back to original item IDs
                            rec_items = item_encoder.inverse_transform(item_indices)
                            als_recommendations[user_id] = list(zip(rec_items, scores))
                        else:
                            als_recommendations[user_id] = []
                    except Exception as e:
                        print(f"Error recommending for user {user_id}: {e}")
                        als_recommendations[user_id] = []
                        
        except Exception as e:
            print(f"Error training ALS: {e}")
            als_recommendations = {}
        
        # 6. Prepare training data for LightGBM
        print("Preparing LightGBM training data...")
        
        # Create positive examples (actual interactions with high engagement)
        positive_samples = train_df[train_df['engagement_score'] > train_df['engagement_score'].median()].copy()
        positive_samples['target'] = 1
        
        # Create negative examples (random non-interactions)
        # We'll sample some negative examples for training
        negative_samples = []
        
        # For each user, sample some items they haven't interacted with
        user_item_pairs = set(zip(train_df['user_id'], train_df['item_id']))
        all_items_set = set(all_items)
        
        # Sample some negative examples
        np.random.seed(42)
        n_negative = min(len(positive_samples), 50000)  # Limit negative samples
        
        for _ in range(n_negative):
            user = np.random.choice(all_users)
            user_interacted = set(train_df[train_df['user_id'] == user]['item_id'])
            available_items = list(all_items_set - user_interacted)
            
            if available_items:
                item = np.random.choice(available_items)
                negative_samples.append({
                    'user_id': user,
                    'item_id': item,
                    'target': 0
                })
        
        negative_df = pd.DataFrame(negative_samples)
        
        # Combine positive and negative samples
        train_samples = pd.concat([
            positive_samples[['user_id', 'item_id', 'target']],
            negative_df
        ], ignore_index=True)
        
        # Add features
        train_samples = train_samples.merge(user_stats, on='user_id', how='left')
        train_samples = train_samples.merge(item_stats, on='item_id', how='left')
        train_samples = train_samples.merge(user_metadata, on='user_id', how='left')
        train_samples = train_samples.merge(item_metadata, on='item_id', how='left')
        
        # Add embedding features if available
        if 'embedding' in item_embeddings.columns:
            # Process embeddings
            embeddings_list = []
            for idx, row in item_embeddings.iterrows():
                emb = row['embedding']
                if isinstance(emb, list):
                    emb_dict = {f'emb_{i}': val for i, val in enumerate(emb)}
                elif isinstance(emb, np.ndarray):
                    emb_dict = {f'emb_{i}': emb[i] for i in range(len(emb))}
                else:
                    continue
                emb_dict['item_id'] = row['item_id']
                embeddings_list.append(emb_dict)
            
            embeddings_df = pd.DataFrame(embeddings_list)
            if not embeddings_df.empty:
                train_samples = train_samples.merge(embeddings_df, on='item_id', how='left')
        
        # Fill missing values
        for col in train_samples.columns:
            if col not in ['user_id', 'item_id', 'target']:
                if train_samples[col].dtype in ['float64', 'int64']:
                    train_samples[col] = train_samples[col].fillna(train_samples[col].median())
                else:
                    train_samples[col] = train_samples[col].fillna(0)
        
        # Select features for training
        feature_cols = [col for col in train_samples.columns 
                       if col not in ['user_id', 'item_id', 'target'] 
                       and train_samples[col].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        print(f"Training LightGBM on {len(train_samples)} samples with {len(feature_cols)} features...")
        
        # Train LightGBM
        lgb_train = lgb.Dataset(
            train_samples[feature_cols],
            train_samples['target']
        )
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'max_depth': 7,
            'min_child_samples': 20
        }
        
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train],
            verbose_eval=False
        )
        
        # 7. Generate final recommendations
        print("Generating final recommendations...")
        
        all_recommendations = []
        
        # Process users in batches to manage memory
        batch_size = 500
        
        for i in range(0, len(all_users), batch_size):
            batch_users = all_users[i:i+batch_size]
            batch_recommendations = []
            
            for user_id in batch_users:
                # Collect candidate items
                candidates = set()
                
                # Add ALS recommendations
                if user_id in als_recommendations and als_recommendations[user_id]:
                    als_items = [item for item, _ in als_recommendations[user_id][:30]]
                    candidates.update(als_items)
                
                # Add popular items
                candidates.update(popular_items[:50])
                
                # Add user's history
                user_history = train_df[train_df['user_id'] == user_id]['item_id'].unique()[:30]
                candidates.update(user_history)
                
                # Convert to list and limit
                candidates = list(candidates)[:200]
                
                if not candidates:
                    candidates = popular_items[:50]
                
                # Prepare candidate features
                candidate_df = pd.DataFrame({
                    'user_id': [user_id] * len(candidates),
                    'item_id': candidates
                })
                
                # Add features
                candidate_df = candidate_df.merge(user_stats, on='user_id', how='left')
                candidate_df = candidate_df.merge(user_metadata, on='user_id', how='left')
                candidate_df = candidate_df.merge(item_stats, on='item_id', how='left')
                candidate_df = candidate_df.merge(item_metadata, on='item_id', how='left')
                
                if 'emb_0' in train_samples.columns:
                    candidate_df = candidate_df.merge(
                        embeddings_df[['item_id'] + [col for col in embeddings_df.columns if col.startswith('emb_')]], 
                        on='item_id', 
                        how='left'
                    )
                
                # Fill missing values
                for col in feature_cols:
                    if col in candidate_df.columns:
                        if candidate_df[col].dtype in ['float64', 'int64']:
                            candidate_df[col] = candidate_df[col].fillna(candidate_df[col].median())
                        else:
                            candidate_df[col] = candidate_df[col].fillna(0)
                    else:
                        candidate_df[col] = 0
                
                # Predict with LightGBM
                if len(candidates) > 0:
                    candidate_df['prediction'] = gbm.predict(candidate_df[feature_cols])
                    
                    # Get top 10 recommendations
                    top_recs = candidate_df.nlargest(10, 'prediction')
                    
                    for _, row in top_recs.iterrows():
                        batch_recommendations.append({
                            'user_id': int(user_id),
                            'recs': int(row['item_id'])
                        })
                else:
                    # Fallback to popular items
                    for item in popular_items[:10]:
                        batch_recommendations.append({
                            'user_id': int(user_id),
                            'recs': int(item)
                        })
            
            all_recommendations.extend(batch_recommendations)
            
            if (i // batch_size) % 5 == 0:
                print(f"Processed {min(i+batch_size, len(all_users))}/{len(all_users)} users")
        
        # Create final submission
        result_df = pd.DataFrame(all_recommendations)
        
        # Ensure we have recommendations for all users
        users_with_recs = result_df['user_id'].unique()
        missing_users = set(all_users) - set(users_with_recs)
        
        if missing_users:
            print(f"Adding recommendations for {len(missing_users)} missing users...")
            for user_id in missing_users:
                for item in popular_items[:10]:
                    result_df = pd.concat([result_df, pd.DataFrame([{
                        'user_id': int(user_id),
                        'recs': int(item)
                    }])], ignore_index=True)
        
        print(f"Saving results to {output_path}...")
        result_df.to_csv(output_path, index=False)
        print(f"Done! Generated {len(result_df)} recommendations for {len(all_users)} users.")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Create empty submission as fallback
        pd.DataFrame(columns=['user_id', 'recs']).to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Recommender System")
    parser.add_argument("--input_path_interactions", type=str, default="train.parquet")
    parser.add_argument("--input_path_users", type=str, default="user_metadata.parquet")
    parser.add_argument("--input_path_items", type=str, default="item_metadata.parquet")
    parser.add_argument("--input_path_embeddings", type=str, default="item_embeddings.parquet")
    parser.add_argument("--output_path", type=str, default="submission.csv")
    
    args = parser.parse_args()
    main(
        args.input_path_interactions,
        args.input_path_users,
        args.input_path_items,
        args.input_path_embeddings,
        args.output_path
    )