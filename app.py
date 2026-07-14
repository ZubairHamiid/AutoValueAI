#http://127.0.0.1:8504 run it on browser

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None


DATA_PATH = Path(__file__).parent / "Data" / "car data.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
]
TARGET_COLUMN = "Selling_Price"


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    mae: float
    rmse: float
    r2: float
    y_test: pd.Series
    y_pred: np.ndarray


def _inject_style() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, #f7f2ea 0%, transparent 45%),
                    radial-gradient(circle at 100% 0%, #e5f3ef 0%, transparent 40%),
                    linear-gradient(180deg, #fffdf8 0%, #f6faf9 100%);
                color: #1d2b2a;
            }

            .block-container {
                padding-top: 2rem;
            }

            .hero {
                border-radius: 18px;
                background: linear-gradient(120deg, #0f766e 0%, #115e59 35%, #1f2937 100%);
                color: #f8fafc;
                padding: 1.4rem 1.2rem;
                box-shadow: 0 16px 35px rgba(17, 94, 89, 0.25);
                margin-bottom: 1rem;
                animation: fadeIn 0.6s ease;
            }

            .metric-card {
                border-radius: 14px;
                background: white;
                border: 1px solid #dce7e4;
                padding: 0.9rem;
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.04);
                animation: riseUp 0.45s ease;
            }

            .small-muted {
                color: #50615f;
                font-size: 0.9rem;
            }

            .stButton > button {
                border-radius: 12px;
                border: 0;
                background: linear-gradient(120deg, #0f766e 0%, #0ea5a4 100%);
                color: white;
                font-weight: 700;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @keyframes riseUp {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    numeric_cols = ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()
    df["Car_Age"] = pd.Timestamp.now().year - df["Year"]
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def build_model(choice: str) -> Pipeline:
    preprocessor = build_preprocessor()

    if choice == "Linear Regression":
        regressor = LinearRegression()
    elif choice == "Random Forest":
        regressor = RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            min_samples_split=3,
            n_jobs=-1,
        )
    else:
        regressor = SVR(kernel="rbf", C=10.0, epsilon=0.2)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


@st.cache_resource(show_spinner=False)
def train_and_evaluate(model_choice: str) -> ModelResult:
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = build_model(model_choice)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return ModelResult(
        name=model_choice,
        model=model,
        mae=mae,
        rmse=rmse,
        r2=r2,
        y_test=y_test,
        y_pred=y_pred,
    )


@st.cache_resource(show_spinner=False)
def run_svr_gridsearch() -> Dict[str, object]:
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", SVR()),
        ]
    )

    # Use only valid SVR kernels and deterministic CV.
    param_grid = {
        "regressor__C": [0.1, 0.5, 1.0, 10.0],
        "regressor__kernel": ["linear", "rbf", "poly"],
        "regressor__degree": [2, 3, 4],
        "regressor__gamma": ["scale", "auto"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best = {
        "C": grid.best_params_["regressor__C"],
        "degree": grid.best_params_["regressor__degree"],
        "kernel": grid.best_params_["regressor__kernel"],
        "gamma": grid.best_params_["regressor__gamma"],
        "score_rmse_cv": float(-grid.best_score_),
    }
    return best


def render_home(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero">
            <h2 style="font-family: 'Space Grotesk', sans-serif; margin: 0;">Used Car Price Intelligence Dashboard</h2>
            <p style="margin-top: .5rem;">Explore market patterns, compare model performance, and get instant selling-price predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><b>Total Cars</b><br><span class='small-muted'>{len(df)}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Avg Price</b><br><span class='small-muted'>{df['Selling_Price'].mean():.2f} lakh</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Max Price</b><br><span class='small-muted'>{df['Selling_Price'].max():.2f} lakh</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><b>Avg Car Age</b><br><span class='small-muted'>{df['Car_Age'].mean():.1f} years</span></div>", unsafe_allow_html=True)

    st.subheader("Market Overview")
    left, right = st.columns([1.2, 1])

    with left:
        fig_price = px.histogram(
            df,
            x="Selling_Price",
            nbins=25,
            title="Selling Price Distribution",
            color_discrete_sequence=["#0f766e"],
        )
        fig_price.update_layout(template="plotly_white")
        st.plotly_chart(fig_price, use_container_width=True)

    with right:
        fuel_avg = df.groupby("Fuel_Type", as_index=False)["Selling_Price"].mean()
        fig_fuel = px.bar(
            fuel_avg,
            x="Fuel_Type",
            y="Selling_Price",
            title="Average Price by Fuel Type",
            color="Fuel_Type",
            color_discrete_sequence=["#0f766e", "#334155", "#eab308"],
        )
        fig_fuel.update_layout(showlegend=False, template="plotly_white")
        st.plotly_chart(fig_fuel, use_container_width=True)


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("Interactive Analytics")

    year_range = st.slider(
        "Filter by Year",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=(int(df["Year"].min()), int(df["Year"].max())),
    )

    filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            filtered,
            x="Present_Price",
            y="Selling_Price",
            color="Fuel_Type",
            size="Driven_kms",
            hover_data=["Car_Name", "Year", "Transmission"],
            title="Selling vs Present Price",
        )
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        trend = filtered.groupby("Year", as_index=False)["Present_Price"].mean()
        fig_trend = px.line(
            trend,
            x="Year",
            y="Present_Price",
            markers=True,
            title="Average Present Price Trend",
            color_discrete_sequence=["#0f766e"],
        )
        fig_trend.update_layout(template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.dataframe(filtered.head(30), use_container_width=True)


def render_model_lab() -> None:
    st.subheader("Model Lab")
    model_choice = st.selectbox("Choose model", ["Linear Regression", "Random Forest", "SVR"])

    if st.button("Train / Evaluate", use_container_width=True):
        result = train_and_evaluate(model_choice)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{result.mae:.4f}")
        m2.metric("RMSE", f"{result.rmse:.4f}")
        m3.metric("R2", f"{result.r2:.4f}")

        chart_df = pd.DataFrame(
            {
                "Actual": result.y_test.values,
                "Predicted": result.y_pred,
            }
        )

        fig_compare = px.scatter(
            chart_df,
            x="Actual",
            y="Predicted",
            title=f"Actual vs Predicted ({result.name})",
            trendline="ols",
            color_discrete_sequence=["#0f766e"],
        )
        fig_compare.update_layout(template="plotly_white")
        st.plotly_chart(fig_compare, use_container_width=True)

        with st.expander("Run SVR Grid Search"):
            if st.button("Find best SVR parameters"):
                best = run_svr_gridsearch()
                st.success("Best SVR configuration found")
                st.json(best)


def render_predict(df: pd.DataFrame) -> None:
    st.subheader("Predict Selling Price")
    st.caption("Enter the car details below to estimate the selling price.")

    with st.expander("What each field means", expanded=True):
        st.markdown(
            """
            - **Orange slider**: the car's model year. Move it left for older cars and right for newer cars.
            - **Prediction Model**: choose the algorithm used to estimate the price.
            - **Owner**: number of previous owners. `0` means first owner, `1` means one previous owner, and higher numbers mean more previous owners.
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Year**")
        year = st.slider(
            "Year of the car",
            int(df["Year"].min()),
            int(df["Year"].max()),
            int(df["Year"].median()),
            help="Model year of the car. The highlighted orange handle is your selected year.",
        )
        st.caption("Orange handle = selected year")
        st.markdown("**Present Price (lakh)**")
        present_price = st.number_input(
            "Present Price (lakh)",
            min_value=float(df["Present_Price"].min()),
            max_value=float(df["Present_Price"].max()),
            value=float(df["Present_Price"].median()),
            help="Current showroom price of the car in lakh.",
        )
        st.markdown("**Driven Kms**")
        driven_kms = st.number_input(
            "Driven Kms",
            min_value=int(df["Driven_kms"].min()),
            max_value=int(df["Driven_kms"].max()),
            value=int(df["Driven_kms"].median()),
            help="Total distance driven by the car in kilometres.",
        )

    with col2:
        st.markdown("**Fuel Type**")
        fuel = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique().tolist()), help="Choose the fuel used by the car.")
        st.markdown("**Selling Type**")
        selling_type = st.selectbox(
            "Selling Type",
            sorted(df["Selling_type"].unique().tolist()),
            help="Dealer means sold through a dealer; Individual means sold directly by the owner.",
        )
        st.markdown("**Transmission**")
        transmission = st.selectbox(
            "Transmission",
            sorted(df["Transmission"].unique().tolist()),
            help="Manual or Automatic gearbox.",
        )
        st.markdown("**Owner**")
        owner_values = sorted(df["Owner"].unique().tolist())

        def owner_label(value: int) -> str:
            if value == 0:
                return "0 - First owner"
            if value == 1:
                return "1 - One previous owner"
            return f"{value} - {value} previous owners"

        owner = st.selectbox(
            "Owner",
            owner_values,
            format_func=owner_label,
            help="This is the number of previous owners for the car.",
        )

    st.markdown("**Prediction Model**")
    model_choice = st.selectbox(
        "Prediction Model",
        ["Linear Regression", "Random Forest", "SVR"],
        help="This chooses the prediction algorithm. Select one from the dropdown instead of using radio dots.",
    )

    if st.button("Predict Price", use_container_width=True):
        result = train_and_evaluate(model_choice)
        input_df = pd.DataFrame(
            [
                {
                    "Year": year,
                    "Present_Price": present_price,
                    "Driven_kms": driven_kms,
                    "Fuel_Type": fuel,
                    "Selling_type": selling_type,
                    "Transmission": transmission,
                    "Owner": owner,
                }
            ]
        )

        pred = result.model.predict(input_df)[0]
        st.success(f"Estimated Selling Price: {pred:.2f} lakh")


def render_api_docs() -> None:
    st.subheader("FastAPI Endpoint")
    st.write(
        "This project also exposes an API endpoint from the same file. "
        "Run the API with: `uvicorn app:api --reload --port 8000`"
    )

    st.code(
        """POST /predict
{
  "Year": 2018,
  "Present_Price": 8.5,
  "Driven_kms": 42000,
  "Fuel_Type": "Petrol",
  "Selling_type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0
}
""",
        language="json",
    )


class CarInput(BaseModel):
    Year: int = Field(..., ge=1990, le=2100)
    Present_Price: float = Field(..., gt=0)
    Driven_kms: float = Field(..., ge=0)
    Fuel_Type: str
    Selling_type: str
    Transmission: str
    Owner: int = Field(..., ge=0, le=5)


def _api_predict(payload: CarInput) -> Dict[str, float]:
    result = train_and_evaluate("Random Forest")
    input_df = pd.DataFrame([payload.model_dump()])
    pred = float(result.model.predict(input_df)[0])
    return {"predicted_selling_price_lakh": round(pred, 4)}


def _build_api() -> Optional[object]:
    if FastAPI is None:
        return None

    app = FastAPI(title="Used Cars Price Prediction API", version="1.0.0")

    @app.get("/")
    def health() -> Dict[str, str]:
        return {"status": "ok", "service": "used-cars-price-api"}

    @app.post("/predict")
    def predict(payload: CarInput) -> Dict[str, float]:
        return _api_predict(payload)

    return app


api = _build_api()
app = api


def main() -> None:
    st.set_page_config(
        page_title="Used Cars Price Dashboard",
        page_icon="🚗",
        layout="wide",
    )
    _inject_style()

    df = load_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Analytics", "Model Lab", "Predict", "API"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit + scikit-learn + FastAPI")

    if page == "Home":
        render_home(df)
    elif page == "Analytics":
        render_analytics(df)
    elif page == "Model Lab":
        render_model_lab()
    elif page == "Predict":
        render_predict(df)
    else:
        render_api_docs()


if __name__ == "__main__":
    main()
