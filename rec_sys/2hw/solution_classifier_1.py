import pandas as pd
import lightgbm as lgb
import gc
import numpy as np

def main():
    # Задаем пути по умолчанию
    input_path_interactions = "interactions.parquet"
    input_path_users = "users.parquet"
    input_path_items = "items.parquet" 
    input_path_embeddings = "embeddings.parquet"
    output_path = "recommendations.csv"
    
    # Загрузка данных
    print("Load data.")
    train_df = pd.read_parquet(input_path_interactions)
    user_metadata = pd.read_parquet(input_path_users)
    item_metadata = pd.read_parquet(input_path_items)
    item_embeddings = pd.read_parquet(input_path_embeddings)

    # Подготовка данных
    item_metadata = item_metadata.merge(item_embeddings, on=["item_id"], how="left")
    train_users = pd.DataFrame({"user_id": train_df["user_id"].unique()})
    train_items = pd.DataFrame({"item_id": train_df["item_id"].unique()})
    
    # Улучшенное определение целевой переменной
    timespent_threshold = train_df["timespent"].quantile(0.7)  # Берем верхние 30%
    train_df["rating"] = (train_df["timespent"] > timespent_threshold).astype(int)
    print(f"Positive rate: {train_df['rating'].mean():.3f}")

    # Решение через LightGBM классификатор
    print("Prepare train dataset.")
    
    # Добавляем embedding фичи если они есть
    embedding_features = [col for col in item_embeddings.columns if col not in ['item_id']]
    features = ["age", "gender", "author_id", "duration"] + embedding_features
    
    train = (train_df
             .merge(user_metadata, on=["user_id"], how="inner")
             .merge(item_metadata, on=["item_id"], how="inner")
             )[["user_id", "item_id"] + features + ["rating"]]

    lgb_train = lgb.Dataset(train[features], train["rating"], 
                           categorical_feature=["age", "gender", "author_id"])

    # Улучшенные параметры
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,  # Увеличено для лучшей выразительности
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'force_row_wise': True  # Для оптимизации памяти
    }

    print("Train classifier.")
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,  # Увеличено количество итераций
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    callbacks=[lgb.log_evaluation(50)]  # Логирование каждые 50 итераций
                    )

    print("Prepare predictions dataframe.")

    # Улучшенная стратегия выборки кандидатов
    item_popularity = (train_df
           .groupby("item_id", as_index=False)["user_id"]
           .count()
           .rename(columns={"user_id": "interaction_count"})
           .sort_values("interaction_count", ascending=False)
           )
    
    # Берем топ популярные + случайные для разнообразия
    top_items = item_popularity.head(100)["item_id"]
    random_items = train_items.sample(frac=0.2, random_state=42)["item_id"]
    sample_items = pd.concat([top_items, random_items]).drop_duplicates()
    
    print(f"Selected {len(sample_items)} candidate items")

    prediction_df = (pd
                     .merge(train_users, sample_items.to_frame(), how="cross")
                     .merge(user_metadata[["user_id", "age", "gender"]], on=["user_id"], how="inner")
                     .merge(item_metadata[["item_id", "author_id", "duration"] + embedding_features], on=["item_id"], how="inner")
                     .merge(train_df[["user_id", "item_id", "rating"]], on=["user_id", "item_id"], how="left")
                     .query("rating.isnull()", engine="python")
                     )

    print("Compute predictions.")
    y_pred = gbm.predict(prediction_df[features], num_iteration=gbm.best_iteration)

    prediction_df["prediction"] = y_pred

    # Более эффективная группировка
    print("Generate recommendations.")
    
    def get_top_recommendations(group):
        return group.nlargest(10, "prediction")["item_id"].tolist()
    
    recommendations = (prediction_df
                      .groupby("user_id")
                      .apply(get_top_recommendations)
                      .reset_index(name='recs'))
    
    result = recommendations.explode("recs")

    print(f"Save result to file {output_path}.")
    result[["user_id", "recs"]].to_csv(output_path, index=False)
    print(f"File saved. Recommendations generated for {len(recommendations)} users.")
    print(f"Average recommendations per user: {result.groupby('user_id').size().mean():.1f}")

if __name__ == "__main__":
    main()