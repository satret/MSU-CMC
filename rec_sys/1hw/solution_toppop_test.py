import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import argparse

def main(input_path: str, output_path: str):
    """
    Обучает модель ALS на данных о взаимодействиях и генерирует файл с рекомендациями.
    """
    print(f"Загрузка данных из {input_path}...")
    try:
        train = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Ошибка при чтении файла {input_path}: {e}")
        return

    print("Подготовка данных для ALS...")
    
    # 1. Отфильтровываем дизлайки, так как ALS лучше работает с позитивными сигналами
    train_als = train[train['dislike'] == False].copy()

    # 2. Создаем "уверенность" (confidence score)
    # 1 балл за просмотр, +4 за лайк (итого 5)
    train_als['confidence'] = 1 + (train_als['like'].astype(int) * 4)

    # 3. Создаем маппинги для user_id и item_id, т.к. implicit требует ID с 0
    # Используем pd.Categorical для эффективного преобразования
    train_als['user_id_cat'] = train_als['user_id'].astype('category')
    train_als['item_id_cat'] = train_als['item_id'].astype('category')

    # Сохраняем маппинги для обратного преобразования
    user_cat_map = dict(enumerate(train_als['user_id_cat'].cat.categories))
    item_cat_map = dict(enumerate(train_als['item_id_cat'].cat.categories))
    
    # Реверсивные маппинги для поиска
    user_id_to_cat = {v: k for k, v in user_cat_map.items()}
    
    print("Создание разреженной матрицы...")
    # user_item_matrix: строки = пользователи, столбцы = объекты
    user_item_matrix = csr_matrix((
        train_als['confidence'].astype(float),
        (train_als['user_id_cat'].cat.codes, train_als['item_id_cat'].cat.codes)
    ))

    print("Обучение модели ALS...")
    model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=20,
        random_state=42,
        use_gpu=False
    )

    # ВАЖНО: обучаем на user_item_matrix, без транспонирования
    model.fit(user_item_matrix)

    print("Генерация рекомендаций...")
    
    # 5. Подготовка пользователей для рекомендации
    # Нам нужны рекомендации для *всех* пользователей из исходного train
    all_original_user_ids = train['user_id'].unique()
    
    # Разделяем пользователей: тех, кто есть в ALS, и тех, кто был отфильтрован (cold start / only dislikes)
    users_in_als_model = set(train_als['user_id'].unique())
    
    users_to_recommend_als = []
    users_fallback_pop = []
    
    for uid in all_original_user_ids:
        if uid in users_in_als_model:
            users_to_recommend_als.append(uid)
        else:
            users_fallback_pop.append(uid)

    results_list = []
    N = 10 # Количество рекомендаций

    # 6. Генерируем рекомендации для пользователей, известных модели
    if users_to_recommend_als:
        print(f"Генерация {len(users_to_recommend_als)} персонализированных рекомендаций...")
        user_codes_to_rec = [user_id_to_cat[uid] for uid in users_to_recommend_als]
        
        # Получаем рекомендации (N=10)
        # model.recommend возвращает (item_codes, scores)
        als_recs_codes, _ = model.recommend(
            userid=user_codes_to_rec,
            user_items=user_item_matrix[user_codes_to_rec],  # то же user_item_matrix
            N=N,
            filter_already_liked_items=True
        )
        
        # Преобразуем item_codes обратно в original item_id
        for i, user_code in enumerate(user_codes_to_rec):
            original_user_id = user_cat_map[user_code]
            for item_code in als_recs_codes[i]:
                original_item_id = item_cat_map[item_code]
                results_list.append((original_user_id, original_item_id))

    # 7. Генерируем Top Popular для cold start / only-dislike пользователей
    if users_fallback_pop:
        print(f"Генерация {len(users_fallback_pop)} фоллбэк-рекомендаций (Top Popular)...")
        # Рассчитываем Top-10 (используем исходный 'train', чтобы учесть все просмотры)
        top_10_items = train.groupby('item_id')['user_id'].count().sort_values(ascending=False).index[:N].to_list()
        
        for user_id in users_fallback_pop:
            for item_id in top_10_items:
                results_list.append((user_id, item_id))

    print("Сохранение результата...")
    # 8. Сохранение в CSV
    final_df = pd.DataFrame(results_list, columns=['user_id', 'recs'])
    
    # Убедимся, что типы данных соответствуют uint32, как в исходных данных
    final_df['user_id'] = final_df['user_id'].astype(np.uint32)
    final_df['recs'] = final_df['recs'].astype(np.uint32)
    
    final_df.to_csv(output_path, index=False)
    print(f"Файл с рекомендациями сохранен в {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommender arguments.")
    parser.add_argument("--input_path", type=str, required=True, help="Input path to train parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to csv with recommendations")

    args = parser.parse_args()
    main(args.input_path, args.output_path)