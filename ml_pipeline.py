import datetime
import os
from itertools import chain, combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import mlflow_utils
from model_training import train_optimize  # noqa: F401


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


def feature_engineering(df, ohe_encoder=None, scale=False, scaler=None):
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

    df.reset_index(drop=True, inplace=True)

    if "Survived" in df.columns:
        survived = df.Survived
    else:
        survived = None

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
    ]

    categorical_cols = df[columns_to_encode].select_dtypes(include=["object"]).columns
    numeric_cols = df[columns_to_encode].select_dtypes(exclude=["object"]).columns

    # Avoid data leakage by fitting the encoder on the training data only
    if ohe_encoder is None:
        ohe_encoder = OneHotEncoder(
            handle_unknown="ignore"
        )  # test data or new incoming data might have categories not seen in the training phase. It ensures that your pipeline is robust to such discrepancies.

        # Fit and transform the selected columns
        ohe_encoder.fit(df[categorical_cols])

    encoded_categorical_data = ohe_encoder.transform(df[categorical_cols])

    encoded_categorical_df = pd.DataFrame(
        encoded_categorical_data.toarray(),
        columns=ohe_encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )

    processed_df = pd.concat(
        [df[numeric_cols], encoded_categorical_df, survived], axis=1
    )

    # Scale data
    if scale:
        # Avoid data leakage by fitting the scaler on the training data only
        if scaler is None:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            scaler.fit(df[["Age", "SibSp", "Parch", "norm_fare"]])

        processed_df[["Age", "SibSp", "Parch", "norm_fare"]] = scaler.transform(
            df[["Age", "SibSp", "Parch", "norm_fare"]]
        )

        return processed_df, ohe_encoder, categorical_cols, scaler

    else:
        return processed_df, ohe_encoder, categorical_cols


def create_feature_groups(df, categorical_cols):
    feature_groups = []
    for cat_col in categorical_cols:
        # Find all OHE columns for this categorical column
        ohe_features = [col for col in df.columns if col.startswith(cat_col + "_")]
        if ohe_features:
            # Group all OHE columns together
            feature_groups.append(ohe_features)
        else:
            # If no OHE columns, just include the original column
            feature_groups.append([cat_col])

    # Add non-categorical columns as individual groups
    non_cat_cols = df.select_dtypes(exclude=["object"]).columns
    for col in non_cat_cols:
        if col not in sum(
            feature_groups, []
        ):  # Avoid duplicating columns that are already grouped
            feature_groups.append([col])

    return feature_groups


def grouped_powerset(feature_groups, include_empty=True):
    if include_empty:
        n = 0
    else:
        n = 1

    return list(
        chain.from_iterable(
            combinations(feature_groups, r) for r in range(n, len(feature_groups) + 1)
        )
    )


def get_features_combinations(df, categorical_cols):
    # combinations = powerset(df.columns, include_empty=False)
    feature_groups = create_feature_groups(df, categorical_cols)

    flattened_feature_groups = [col for sublist in feature_groups for col in sublist]
    if not set(flattened_feature_groups) == set(df.columns):
        raise ValueError(
            "Feature groups do not cover all columns in the dataframe. Please check the feature groups."
        )

    grouped_combinations = grouped_powerset(feature_groups, include_empty=False)

    return grouped_combinations


def ml_pipeline():
    # Train model
    dataset = get_dataset(data="train")

    # numeric_cols = ["Age", "SibSp", "Parch", "Fare"]
    # categorial_cols = ["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]

    dataset, age_median, fare_median = data_cleaning(dataset)

    scale = False
    results = feature_engineering(dataset, scale=scale)
    if scale:
        dataset, ohe_encoder, categorical_cols, scaler = (
            results[0],
            results[1],
            results[2],
            results[3],
        )
    else:
        scaler = None
        dataset, ohe_encoder, categorical_cols = results[0], results[1], results[2]

    target = "Survived"
    y = dataset[target]
    X = dataset.drop([target], axis=1)

    # best_estimator, best_params, best_score = train_optimize(X, y)

    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)

    features_combinations = get_features_combinations(X, categorical_cols)

    experiment_name = "Titanic"
    mlflow_tracking_uri = "http://127.0.0.1:5000"

    mlflow_tracking_uri = mlflow_utils.start_mlflow_server()

    print(
        f"MLflow server is running at: {mlflow_tracking_uri}, Experiment: {experiment_name}"
    )

    experiment_id, mlflow_client = mlflow_utils.set_mlflow(
        experiment_name, mlflow_tracking_uri=mlflow_tracking_uri
    )

    return best_estimator, age_median, fare_median, ohe_encoder, scaler


def predict_pipeline(best_estimator, age_median, fare_median, ohe_encoder, scaler):
    # Predict on test data
    dataset = get_dataset(data="test")
    passenger_id = dataset.PassengerId

    dataset, _, _ = data_cleaning(
        dataset, age_median=age_median, fare_median=fare_median
    )

    if scaler is not None:
        scale = True
    else:
        scale = False

    results = feature_engineering(
        dataset, ohe_encoder=ohe_encoder, scale=scale, scaler=scaler
    )
    dataset, ohe_encoder = results[0], results[1]

    X = dataset

    y_pred = best_estimator.predict(X)

    output = pd.DataFrame({"PassengerId": passenger_id, "Survived": y_pred})

    return best_estimator, output


if __name__ == "__main__":
    best_estimator, age_median, fare_median, ohe_encoder, scaler = ml_pipeline()

    model, y_pred_df = predict_pipeline(
        best_estimator, age_median, fare_median, ohe_encoder, scaler
    )

    submission_folder = "Submissions"
    os.makedirs(submission_folder, exist_ok=True)

    filename = (
        f"submission_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}.csv"
    )

    path = os.path.join(submission_folder, filename)

    y_pred_df.to_csv(path, index=False)
