import datetime
import os
import warnings
from itertools import chain, combinations
from typing import Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import ray
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow_utils
from model_training import train_optimize  # noqa: F401

# check if the issue was fixed in the latest version of the library: https://github.com/mlflow/mlflow/issues/10709
warnings.filterwarnings("ignore", category=FutureWarning)


def get_dataset(data: Literal["train", "test"] = "train") -> pd.DataFrame:
    data_folder = "Data"

    if data == "train":
        filename = "train.csv"

    else:
        filename = "test.csv"

    path = os.path.join(data_folder, filename)
    df = pd.read_csv(path)

    return df


def data_cleaning(
    df: pd.DataFrame,
    age_median: Optional[float] = None,
    fare_median: Optional[float] = None,
) -> Tuple[pd.DataFrame, float, float]:
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


def feature_engineering(
    df: pd.DataFrame,
    ohe_encoder: OneHotEncoder = None,
    scale: bool = False,
    scaler: StandardScaler = None,
) -> Tuple[pd.DataFrame, OneHotEncoder, list[str], StandardScaler]:
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
            scaler = StandardScaler()

            scaler.fit(df[["Age", "SibSp", "Parch", "norm_fare"]])

        processed_df[["Age", "SibSp", "Parch", "norm_fare"]] = scaler.transform(
            df[["Age", "SibSp", "Parch", "norm_fare"]]
        )

        return processed_df, ohe_encoder, categorical_cols, scaler

    else:
        return processed_df, ohe_encoder, categorical_cols


def create_feature_groups(
    df: pd.DataFrame, categorical_cols: list[str]
) -> list[list[str]]:
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


def grouped_powerset(
    feature_groups: list[list[str]], include_empty: bool = True
) -> list[tuple[list[str]]]:
    if include_empty:
        n = 0
    else:
        n = 1

    return list(
        chain.from_iterable(
            combinations(feature_groups, r) for r in range(n, len(feature_groups) + 1)
        )
    )


def get_features_combinations(
    df: pd.DataFrame, categorical_cols: list[str]
) -> list[tuple[list[str]]]:
    # combinations = powerset(df.columns, include_empty=False)
    feature_groups = create_feature_groups(df, categorical_cols)

    flattened_feature_groups = [col for sublist in feature_groups for col in sublist]
    if not set(flattened_feature_groups) == set(df.columns):
        raise ValueError(
            "Feature groups do not cover all columns in the dataframe. Please check the feature groups."
        )

    grouped_combinations = grouped_powerset(feature_groups, include_empty=False)

    return grouped_combinations


def validation(
    features_combination: tuple[list[str]],
    target: str,
    dataset: pd.DataFrame,
    params: dict,
    experiment_id: int,
    mlflow_client: MlflowClient,
) -> None:
    if not isinstance(features_combination, tuple):
        raise ValueError("combination must be a tuple")
    if not isinstance(target, str):
        raise ValueError("target must be a string")
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("dataset must be a pandas DataFrame")
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")
    if not isinstance(experiment_id, Optional[Union[int, str]]):
        raise ValueError("experiment_id must be an integer")
    if not isinstance(mlflow_client, MlflowClient):
        raise ValueError("mlflow_client must be an MlflowClient object")


def features_selection(
    features_combination: tuple[list[str]],
) -> list[str]:
    features = [feature for sublist in features_combination for feature in sublist]

    return features


def clean_finished_combinations(
    combinations: list[dict], experiment_name: str
) -> list[tuple[list[str]]]:
    finished_configs = mlflow_utils.get_finished_configs(experiment_name)

    skipped_configs = [
        combination for combination in combinations if combination in finished_configs
    ]

    if len(skipped_configs) > 0:
        print(f"Skipped {len(skipped_configs)} configurations")

    # Getting the cleaned combinations that are not in finished_configs
    cleaned_combinations = [
        combination
        for combination in combinations
        if combination not in finished_configs
    ]

    return cleaned_combinations


@ray.remote
def train(
    features_combination: tuple[list[str]],
    target: str,
    dataset: pd.DataFrame,
    params: dict,
    experiment_id: int,
    mlflow_client: MlflowClient,
) -> None:
    validation(
        features_combination,
        target,
        dataset,
        params,
        experiment_id,
        mlflow_client,
    )

    run = mlflow_client.create_run(experiment_id)
    run_id = run.info.run_id

    try:
        mlflow_client.log_param(run_id, "features", features_combination)

        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.autolog()

            X = dataset[features_selection(features_combination)]
            y = dataset[target]

            model = RandomForestClassifier(**params)

            model.fit(X, y)  # MLflow triggers logging automatically upon model fitting

        mlflow_client.set_terminated(
            run_id, status=RunStatus.to_string(RunStatus.FINISHED)
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        mlflow_client.set_terminated(
            run_id, status=RunStatus.to_string(RunStatus.FAILED)
        )
        raise  # Re-raise the exception to mark run as failed

    except KeyboardInterrupt:
        # If the run is interrupted (e.g., by pressing Ctrl+C), log it as KILLED
        print("Run interrupted by user.")
        mlflow_client.set_terminated(
            run_id, status=RunStatus.to_string(RunStatus.KILLED)
        )
        raise  # Re-raise the exception to mark run as failed

    else:
        # If everything goes well, the run will be logged as FINISHED
        pass


def ml_pipeline() -> (
    Tuple[RandomForestClassifier, float, float, OneHotEncoder, StandardScaler]
):
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

    # best_estimator, best_params, best_score = train_optimize(X, y)   # TODO
    best_params = {
        "bootstrap": True,
        "criterion": "entropy",
        "max_depth": 15,
        "min_samples_leaf": 2,
        "min_samples_split": 15,
        "n_estimators": 100,
        "random_state": 42,
    }

    # print("Best Parameters:", best_params)
    # print("Best Score:", best_score)

    features_combinations = get_features_combinations(X, categorical_cols)

    experiment_name = "Titanic"

    experiment_id, mlflow_client = mlflow_utils.set_mlflow(
        experiment_name, mlflow_tracking_uri=mlflow_utils.start_mlflow_server()
    )

    # TODO remove trim (testing purposes)
    features_combinations = features_combinations[:10]

    features_combinations = clean_finished_combinations(
        features_combinations, experiment_name
    )

    ray.init()

    result_ids = [
        train.remote(
            combination,
            target=target,
            dataset=dataset,
            params=best_params,
            experiment_id=experiment_id,
            mlflow_client=mlflow_client,
        )
        for combination in features_combinations
    ]

    results = ray.get(result_ids)

    ray.shutdown()

    return best_estimator, age_median, fare_median, ohe_encoder, scaler


def predict_pipeline(
    best_estimator: RandomForestClassifier,
    age_median: Optional[float],
    fare_median: Optional[float],
    ohe_encoder: OneHotEncoder,
    scaler: StandardScaler,
) -> (RandomForestClassifier, pd.DataFrame):
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
