
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


NUMERICAL_FEATURES = [
    "Sales",
    "Quantity",
    "Discount",
    "Shipping Cost",
    "Order Year",
    "Order Month"
]

CATEGORICAL_FEATURES = [
    "Ship Mode",
    "Segment",
    "City",
    "State",
    "Country",
    "Market",
    "Region",
    "Category",
    "Sub-Category",
    "Order Priority"
]


def build_preprocessor(strategy):

    if strategy == "standard":
        scaler = StandardScaler()

    elif strategy == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("Unknown preprocessing strategy!")

    numerical_transformer = scaler

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, NUMERICAL_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor
