import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from term_colours import TermColours as tc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sb


def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))


#
# FFMC: Fine Fuel Moisture Code
#   It is intended to represent moisture conditions for shaded litter fuels,
#   the equivalent of 16-hour timelag
#
# DMC: Duff Moisture Code
#   represents fuel moisture of decomposed organic material underneath the litter.
#
# DC: Drought Code
#   represents drying deep into the soil.
#
# ISI: Initial Spread Index
#   It integrates fuel moisture for fine dead fuels and surface windspeed to estimate a spread potential.
#

if __name__ == '__main__':
    df = pd.read_csv("data/forestfires.csv")
    tc.info(f"Dataset shape: {df.shape}")
    tc.info("Dataset:")
    print(df.head())
    tc.info("Handling non-numeric data in the dataset...")
    handle_non_numeric_data(df)
    tc.info("Dataset:")

    features = df.drop("area", axis=1)
    labels = df["area"]

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, train_size=0.7,
                                                                                random_state=1)
    fig, ax = plt.subplots(figsize=(10, 10))

    corr = df.corr()
    sb.heatmap(corr.iloc[:, :], cmap="Blues", annot=True)

    plt.show()

    standard_scaler = StandardScaler()
    standard_scaler.fit(features_train)

    features_train = pd.DataFrame(standard_scaler.transform(features_train), columns=features.columns)
    features_test = pd.DataFrame(standard_scaler.transform(features_test), columns=features.columns)

    print(features_train.head())
    print("Variance:")
    print(features_train.var())
    print("Mean:")
    print(features_train.mean())

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=12, activation="relu", input_shape=[12]),
        tf.keras.layers.Dense(units=12, activation="relu"),
        tf.keras.layers.Dense(units=8, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="relu")
    ])

    model.summary()

    early_stop = EarlyStopping(monitor="accuracy", mode='max', verbose=1, patience=500)

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy", "mean_squared_error"])
    model.fit(features_train, labels_train, epochs=10000, callbacks=[early_stop])

    test_loss = model.evaluate(features_test, labels_test)

    print(f"Accuracy: {test_loss[1] * 100}")

    model.save("saved_model", save_format="tf")
