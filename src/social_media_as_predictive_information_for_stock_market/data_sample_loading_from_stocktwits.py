import os
import pandas as pd


# Directory where CSV files are located
base_dir = '/home/hachani/Documents/Cours/UE/IA_717 (Natural Language Processing)/NLP_Project/Social-Media-as-Predictive-Information-for-Stock-Market/data'

def formater_matricule(number, folder):
    if folder in ["feature_wo_messages", "messages", "symbols"]:
        threshold = 999
        d=len(str(threshold))
    elif folder in ["msg_info", "sentiments", "symbol_sentiments"]:
        threshold = 99
        d=len(str(threshold))

    if number > threshold:
        raise ValueError(f"Erreur : Le nombre ne doit pas être supérieur à {threshold}.")
    
    return f"{number:0{d}d}"

def path_to_csv(number, folder, dir=base_dir):
    folders = {"feature_wo_messages" : "feature_wo_messages/feature_wo_messages_", 
               "messages" : "messages/msg_", 
               "msg_info" : "msg_info/msg_info_",
               "sentiments" : "sentiments/sentiment_",
               "symbols" : "symbols/symbol_",
               "symbol_sentiments" : "symbol_sentiments/symbol_sentiments_"}
    directory = os.path.join(dir, folders[folder])
    path = directory+formater_matricule(number=number, folder=folder)+".csv"

    return path

def load_csv(number, folder, dir=base_dir, dtype=None):
    matricule = formater_matricule(number, folder)
    path = path_to_csv(number, folder, dir=dir)
    if dtype==None:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, dtype=dtype)

    print(f"folder : {folder}")
    print(f"matricule : {matricule}")
    print(f"path : {path}")

    return df

def is_int_like(val):
    if isinstance(val, int):
        return True
    if isinstance(val, str):
        # Vérifie si la chaîne est un nombre entier (ex: "123" mais pas "123.45" ou "abc")
        return val.isdigit()
    return False

def message_id_cleaning(df):
    df = df[df["message_id"].apply(is_int_like)]
    df["message_id"] = df["message_id"].astype(int)

    return df

def message_id_unique(df):
    df = df.sort_values("message_id")
    df = df.drop_duplicates(subset=["message_id"], keep=False)

    return df

def data_loading(dir=base_dir): 
    num = 0
    folder_sentiment = "feature_wo_messages" 
    df_sentiment = load_csv(number=num, folder=folder_sentiment, dir=dir)
    df_sentiment = df_sentiment.sort_values("message_id")
    df_sentiment["message_id"] = df_sentiment["message_id"].astype("category")
    df_sentiment["user_id"] = df_sentiment["message_id"].astype("category")
    df_sentiment["created_at"] = pd.to_datetime(df_sentiment["created_at"], unit='s')
    df_sentiment["timestamp"] = pd.to_datetime(df_sentiment["created_at"]).astype('int64') // 10**9
    df_sentiment["parent_message_id"] = df_sentiment["parent_message_id"].astype("Int64").astype("category")
    df_sentiment["in_reply_to_message_id"] = df_sentiment["in_reply_to_message_id"].astype("Int64").astype("category")

    folder_message_body = "messages"
    df_message_body = load_csv(number=num, folder=folder_message_body, dir=dir, dtype={"message_id": str})
    df_message_body = message_id_cleaning(df_message_body)
    df_message_body = message_id_unique(df_message_body)
    df_message_body["message_id"] = df_message_body["message_id"].astype("category")

    df = pd.merge(df_message_body, df_sentiment[["message_id", "created_at", "timestamp", "sentiment"]], on="message_id", how="inner")
    df.dropna(subset=["sentiment"], inplace=True)
    df["sentiment_01"] = (df["sentiment"]=="Bullish")*1
    df["sentiment_01"] = df["sentiment_01"].astype("category")
    df["message_id"] = df["message_id"].astype("category")
    del df_message_body
    del df_sentiment

    return df

def data_sampled(df, N_samples=10000, ratio=0.5):
    # Number of samples for each class
    N_samples_bearish = int(N_samples*ratio)
    N_samples_bullish = N_samples - N_samples_bearish
    # Split bullish data from bearish one
    df_bearish = df[df["sentiment_01"] == 0]
    df_bullish = df[df["sentiment_01"] == 1]
    # Data sampled for each class
    df_bearish_sample = df_bearish.sample(n=N_samples_bearish, random_state=42)
    df_bullish_sample = df_bullish.sample(n=N_samples_bullish, random_state=42)
    del df_bearish
    del df_bullish
    # Data sampled
    df_sample = pd.concat([df_bearish_sample, df_bullish_sample])
    df_sample = df_sample.sample(frac=1, random_state=42).reset_index( drop=True)
    del df_bearish_sample
    del df_bullish_sample

    return df_sample