from lib.advisor import Advisor
from dotenv import dotenv_values
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import pandas as pd

config = dotenv_values(".env")

if __name__ == '__main__':
    # тут можно загрузить модель или создать новую
    # advisor = Advisor().load(f"{config['store_models']}/test.bin")
    advisor = Advisor(
        vectorizer=TfidfVectorizer(max_features=10000),
        model=SVC(random_state=int(config['RANDOM_STATE']))
    )

    data = pd.read_csv(f"{config['datasets']}/train_all_unique_re.csv").drop('Unnamed: 0', axis=1)
    data.columns = ['text', 'label']

    train = data.head(1000)
    test = data.tail(1000)

    # подготавлеваем векторизатор
    advisor.fit_vectorizer(data.text)
    # обучаем модель
    advisor.fit_model(train.text, train.label)

    # проверяем скор можно сделать одну метрику
    score = advisor.classification_report(test.text, test.label)

    # возвращаем следующий чанк
    new_chunk = advisor.predict_proba(test.text).head(int(config['chunk_size']))

    print(score)

    # тут можно сохранить модель со всеми весами
    # advisor.save(f"conf['models']test.bin")
