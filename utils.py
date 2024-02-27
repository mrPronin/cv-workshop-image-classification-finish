# utils.py


def encode_labels(df, label_mapping):
    df["encode_label"] = df["label"].map(label_mapping)
    return df


def map_labels(labels):
    return {label: idx for idx, label in enumerate(labels)}
