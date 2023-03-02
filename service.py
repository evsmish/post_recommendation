import os
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet
from datetime import datetime
from sqlalchemy import create_engine
from psycopg2.extensions import register_adapter, AsIs

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)

app = FastAPI()

def batch_load_sql(query: str):
    CHUNKSIZE = 200000
    engine = create_engine(
        'There must be login and password to access server'
    )
    conn = engine.connect().execution_options(
        stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    if os.environ.get("SERVICE_URL") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_features(posts_path: str, users_path: str):
    liked_posts_query = """
    SELECT distinct post_id, user_id
    FROM public.feed_data
    WHERE action = 'like'
    """
    liked_posts = batch_load_sql(liked_posts_query)

    engine = create_engine(
        "There must be login and password to access server"
    )

    post_features = pd.read_sql(posts_path, con=engine)
    user_features = pd.read_sql(users_path, con=engine)

    return [liked_posts, post_features, user_features]

def load_models(mod_path):
    model_path = get_model_path(mod_path)
    model = CatBoostClassifier()
    model = model.load_model(model_path)
    return model

# loading model
model= load_models('There must be model name')
# loading features
features = load_features('post data path', 'user data path')

def get_recommended_feed(id: int, time: datetime, limit: int):
    # extracting features for certain user
    user_features = features[2].loc[features[2]['user_id'] == id]

    # feature preparation for model
    user_features = user_features.drop(['user_id'], axis=1)

    posts_features = features[1].drop(['index'], axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    add_user_features = dict(zip(user_features.columns, user_features.values[0]))

    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    user_posts_features['hour'] = time.hour
    user_posts_features['month'] = time.month

    user_posts_features = user_posts_features.drop(['index', 'text'], axis=1)

    # model prediction
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predicts'] = predicts

    # remove posts that user has already seen/liked
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts['user_id'] == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    # final recommendation
    recommended_posts = filtered_.sort_values('predicts')[-limit:].index

    return [PostGet(**{
        "id": i,
        "text": content[content.post_id == i].text.values[0],
        "topic": content[content.post_id == i].topic.values[0]
    }) for i in recommended_posts
            ]


@app.get("/post/recommendations/")
def recommended_posts(id: int, time: datetime, limit: int = 5):
    return get_recommended_feed(id, time, limit)
