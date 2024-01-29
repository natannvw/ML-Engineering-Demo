import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from model_training import train_optimize


def get_dataset(data="train"):
    data_folder = "Data"

    if data == "train":
        filename = "train.csv"

    else:
        filename = "test.csv"

    path = os.path.join(data_folder, filename)
    df = pd.read_csv(path)

    return df


def data_cleaning(df, age_median=None, fare_median=None):
    # impute nulls for continuous data
    if age_median is None:
        age_median = df.Age.median()
    if fare_median is None:
        fare_median = df.Fare.median()

    df.Age = df.Age.fillna(age_median)
    df.Fare = df.Fare.fillna(fare_median)

    # drop null 'embarked' rows. Only 2 instances of this in training and 0 in test
    df.dropna(subset=["Embarked"], inplace=True)

    return df, age_median, fare_median


def feature_engineering(df, scale=False):
    df["cabin_multiple"] = df.Cabin.apply(
        lambda x: 0 if pd.isna(x) else len(x.split(" "))
    )
    df["cabin_adv"] = df.Cabin.apply(lambda x: str(x)[0])

    df["numeric_ticket"] = df.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
    df["ticket_letters"] = df.Ticket.apply(
        lambda x: "".join(x.split(" ")[:-1]).replace(".", "").replace("/", "").lower()
        if len(x.split(" ")[:-1]) > 0
        else 0
    )

    df["name_title"] = df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

    # log norm of fare
    df["norm_fare"] = np.log(df.Fare + 1)

    # converted fare to category for pd.get_dummies()
    df.Pclass = df.Pclass.astype(str)

    # Select the columns for one-hot encoding
    columns_to_encode = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "norm_fare",
        "Embarked",
        "cabin_adv",
        "cabin_multiple",
        "numeric_ticket",
        "name_title",
        "train_test",
    ]

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the selected columns
    encoded_columns = encoder.fit_transform(df[columns_to_encode]).toarray()

    # Create a DataFrame with the encoded columns
    encoded_df = pd.DataFrame(
        encoded_columns, columns=encoder.get_feature_names(columns_to_encode)
    )

    # Concatenate the encoded DataFrame with the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    # Drop the original columns that were encoded
    # all_dummies.drop(columns_to_encode, axis=1, inplace=True)

    # Scale data
    if scale:
        from sklearn.preprocessing import StandardScaler

        scale = StandardScaler()
        df[["Age", "SibSp", "Parch", "norm_fare"]] = scale.fit_transform(
            df[["Age", "SibSp", "Parch", "norm_fare"]]
        )

    return df


def ml_pipeline():
    dataset = get_dataset(data="train")

    numeric_cols = ["Age", "SibSp", "Parch", "Fare"]
    categorial_cols = ["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]

    dataset, age_median, fare_median = data_cleaning(dataset)

    dataset = feature_engineering(dataset, scale=False)  # TODO: scale=True

    y = dataset["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(dataset[features])

    best_parameters, best_score = train_optimize(X, y)

    print("Best Parameters:", best_parameters)
    print("Best Score:", best_score)


if __name__ == "__main__":
    ml_pipeline()
