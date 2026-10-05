import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="eThekwini 2026 Local Election Analysis",
    page_icon="🗳️",
    layout="wide"
)

st.title("🗳️ eThekwini 2026 Local Election ML Analysis")
st.caption("South African Local Government Elections — eThekwini Metropolitan Municipality")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df_2016 = pd.read_csv("ETH_Ward (2016).csv")
    df_2021 = pd.read_csv("ETH_Ward (2021).csv")

    df_2016.columns = df_2016.columns.str.strip()
    df_2021.columns = df_2021.columns.str.strip()

    for df in [df_2016, df_2021]:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

    return df_2016, df_2021


df_2016, df_2021 = load_data()

# ============================================================
# PREPROCESSING
# ============================================================

df_2016_ward = df_2016[
    df_2016["BallotType"].str.lower() == "ward"
].copy()

df_2021_ward = df_2021[
    df_2021["BallotType"].str.lower() == "ward"
].copy()


# ============================================================
# CREATE WARD DATA
# ============================================================

def create_ward_data(df, year):

    party_votes = (
        df.groupby(["Ward", "PartyName"], as_index=False)
        .agg(Votes=("TotalValidVotes", "sum"))
    )

    ward_info = (
        df[
            [
                "Ward",
                "VotingDistrict",
                "RegisteredVoters",
                "SpoiltVotes"
            ]
        ]
        .drop_duplicates(["Ward", "VotingDistrict"])
        .groupby("Ward", as_index=False)
        .agg(
            RegisteredVoters=("RegisteredVoters", "sum"),
            SpoiltVotes=("SpoiltVotes", "sum")
        )
    )

    ward_data = party_votes.merge(
        ward_info,
        on="Ward",
        how="left"
    )

    ward_data["ElectionYear"] = year

    ward_data["WardValidVotes"] = (
        ward_data.groupby("Ward")["Votes"]
        .transform("sum")
    )

    ward_data["VoteShare"] = (
        ward_data["Votes"] /
        ward_data["WardValidVotes"]
    )

    return ward_data


ward_2016 = create_ward_data(df_2016_ward, 2016)
ward_2021 = create_ward_data(df_2021_ward, 2021)


# ============================================================
# GEOGRAPHIC ALIGNMENT
# ============================================================

def ward_vd_sets(df):

    return (
        df.groupby("Ward")["VotingDistrict"]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )


vd_2016 = ward_vd_sets(df_2016_ward)
vd_2021 = ward_vd_sets(df_2021_ward)

common_wards = sorted(
    set(vd_2016.keys()) &
    set(vd_2021.keys())
)

stable_wards = [
    ward for ward in common_wards
    if vd_2016[ward] == vd_2021[ward]
]

stable_wards = sorted(stable_wards)


# ============================================================
# WARD FEATURE TABLE
# ============================================================

selected_parties = [
    "ANC",
    "DA",
    "EFF",
    "INDEPENDENT",
    "IFP"
]


def make_party_table(ward_data):

    table = ward_data[
        ward_data["PartyName"].isin(selected_parties)
    ].pivot_table(
        index="Ward",
        columns="PartyName",
        values="VoteShare",
        aggfunc="sum",
        fill_value=0
    )

    for party in selected_parties:
        if party not in table.columns:
            table[party] = 0

    return table[selected_parties]


party_2016 = make_party_table(ward_2016)
party_2021 = make_party_table(ward_2021)


# ============================================================
# TURNOUT
# ============================================================

def create_turnout(df):

    turnout = (
        df[
            [
                "Ward",
                "VotingDistrict",
                "RegisteredVoters",
                "SpoiltVotes",
                "TotalValidVotes"
            ]
        ]
        .drop_duplicates(["Ward", "VotingDistrict"])
        .groupby("Ward", as_index=False)
        .agg(
            RegisteredVoters=(
                "RegisteredVoters",
                "sum"
            ),
            SpoiltVotes=(
                "SpoiltVotes",
                "sum"
            ),
            ValidVotes=(
                "TotalValidVotes",
                "sum"
            )
        )
    )

    turnout["Turnout"] = (
        turnout["ValidVotes"] +
        turnout["SpoiltVotes"]
    ) / turnout["RegisteredVoters"]

    return turnout


turnout_2016 = create_turnout(df_2016_ward)
turnout_2021 = create_turnout(df_2021_ward)


# ============================================================
# MODELLING DATA
# ============================================================

model_data = pd.DataFrame(index=stable_wards)

# Make sure every stable ward exists in both election tables
party_2016_aligned = party_2016.reindex(
    stable_wards,
    fill_value=0
)

party_2021_aligned = party_2021.reindex(
    stable_wards,
    fill_value=0
)

for party in selected_parties:

    model_data[party] = party_2016_aligned[party]

    model_data[f"{party}_2021"] = party_2021_aligned[party]

model_data["Turnout_2016"] = turnout_2016.set_index(
    "Ward"
).loc[stable_wards, "Turnout"]

model_data["Turnout_2021"] = turnout_2021.set_index(
    "Ward"
).loc[stable_wards, "Turnout"]

model_data["RegisteredVoters_2016"] = turnout_2016.set_index(
    "Ward"
).loc[stable_wards, "RegisteredVoters"]


# ============================================================
# TRAIN MODELS
# ============================================================

features = [
    "ANC",
    "DA",
    "EFF",
    "INDEPENDENT",
    "IFP",
    "Turnout_2016",
    "RegisteredVoters_2016"
]

targets = [
    "ANC_2021",
    "DA_2021",
    "EFF_2021",
    "INDEPENDENT_2021",
    "IFP_2021",
    "Turnout_2021"
]

X = model_data[features]

models = {}
cv_results = {}

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for target in targets:

    y = model_data[target]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }
    )

    cv_results[target] = {
        "MAE": -scores["test_mae"].mean(),
        "RMSE": -scores["test_rmse"].mean(),
        "R2": scores["test_r2"].mean()
    }

    model.fit(X, y)

    models[target] = model


# ============================================================
# 2026 PREDICTIONS
# ============================================================

latest_2021 = pd.DataFrame(index=sorted(party_2021.index))

for party in selected_parties:
    latest_2021[party] = party_2021.loc[
        latest_2021.index,
        party
    ]

# Use the 2021 election as the latest available state
# for generating the 2026 projection.

latest_2021["Turnout_2016"] = turnout_2021.set_index(
    "Ward"
).reindex(latest_2021.index)["Turnout"]

latest_2021["RegisteredVoters_2016"] = turnout_2021.set_index(
    "Ward"
).reindex(latest_2021.index)["RegisteredVoters"]


prediction_features = latest_2021[
    features
].copy()

predictions = pd.DataFrame(
    index=latest_2021.index
)

for party in selected_parties:

    predictions[party] = models[
        f"{party}_2021"
    ].predict(prediction_features)

    predictions[party] = predictions[party].clip(
        0,
        1
    )


predictions["Turnout"] = models[
    "Turnout_2021"
].predict(prediction_features)

predictions["Turnout"] = predictions[
    "Turnout"
].clip(0, 1)


# ============================================================
# 2026 AGGREGATE PROJECTION
# ============================================================

registered_2021 = (
    df_2021_ward[
        [
            "Ward",
            "VotingDistrict",
            "RegisteredVoters"
        ]
    ]
    .drop_duplicates(["Ward", "VotingDistrict"])
    .groupby("Ward")["RegisteredVoters"]
    .sum()
)

registered_2021 = registered_2021.reindex(
    predictions.index
)

projected_turnout_voters = (
    predictions["Turnout"] *
    registered_2021
)

aggregate = {}

for party in selected_parties:

    projected_votes = (
        predictions[party] *
        projected_turnout_voters
    ).sum()

    aggregate[party] = projected_votes


aggregate_df = pd.DataFrame(
    {
        "Party": list(aggregate.keys()),
        "ProjectedVotes": list(aggregate.values())
    }
)

total_projected_votes = aggregate_df[
    "ProjectedVotes"
].sum()

aggregate_df["ProjectedShare"] = (
    aggregate_df["ProjectedVotes"] /
    total_projected_votes
)

aggregate_df = aggregate_df.sort_values(
    "ProjectedVotes",
    ascending=False
).reset_index(drop=True)


total_registered = registered_2021.sum()

projected_turnout_voters_total = (
    projected_turnout_voters.sum()
)

projected_turnout_rate = (
    projected_turnout_voters_total /
    total_registered
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard")

page = st.sidebar.radio(
    "Select page",
    [
        "Overview",
        "Historical Analysis",
        "2026 Model Outputs",
        "Selected Wards",
        "Model Evaluation",
        "Uncertainty & Limitations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Overview")

    st.write(
        """
        This dashboard presents an evidence-based machine learning
        analysis for the 2026 South African Local Government Elections
        in eThekwini Metropolitan Municipality.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "2016 wards",
        df_2016_ward["Ward"].nunique()
    )

    col2.metric(
        "2021 wards",
        df_2021_ward["Ward"].nunique()
    )

    col3.metric(
        "Stable wards",
        len(stable_wards)
    )

    col4.metric(
        "2026 projected turnout",
        f"{projected_turnout_rate:.2%}"
    )

    st.subheader("Projected 2026 Aggregate Result")

    st.dataframe(
        aggregate_df,
        use_container_width=True
    )

    leader = aggregate_df.iloc[0]

    st.success(
        f"Projected leading party: {leader['Party']} "
        f"with approximately {leader['ProjectedShare']:.2%} "
        f"of the projected votes."
    )

    st.info(
        "No single relevant party is projected to obtain a majority "
        "of the projected vote share. Coalition formation may therefore "
        "be required to reach a majority. The model does not predict "
        "which parties would form a governing coalition."
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

elif page == "Historical Analysis":

    st.header("Historical Analysis")

    st.subheader("2016 Party Vote Share")

    historical_2016 = (
        party_2016.mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    historical_2016.columns = [
        "Party",
        "AverageVoteShare"
    ]

    historical_2016["AverageVoteShare"] = (
        historical_2016["AverageVoteShare"] * 100
    )

    st.dataframe(
        historical_2016,
        use_container_width=True
    )

    st.subheader("2021 Party Vote Share")

    historical_2021 = (
        party_2021.mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    historical_2021.columns = [
        "Party",
        "AverageVoteShare"
    ]

    historical_2021["AverageVoteShare"] = (
        historical_2021["AverageVoteShare"] * 100
    )

    st.dataframe(
        historical_2021,
        use_container_width=True
    )

    st.subheader("Historical Turnout")

    turnout_comparison = pd.DataFrame({
        "Year": [2016, 2021],
        "AverageTurnout": [
            turnout_2016["Turnout"].mean(),
            turnout_2021["Turnout"].mean()
        ]
    })

    st.line_chart(
        turnout_comparison.set_index("Year")
    )


# ============================================================
# 2026 MODEL OUTPUTS
# ============================================================

elif page == "2026 Model Outputs":

    st.header("2026 Model Outputs")

    st.subheader("Projected Party Results")

    display_aggregate = aggregate_df.copy()

    display_aggregate["ProjectedShare"] = (
        display_aggregate["ProjectedShare"] * 100
    )

    st.dataframe(
        display_aggregate,
        use_container_width=True
    )

    st.subheader("Projected Vote Share")

    chart_data = display_aggregate.set_index(
        "Party"
    )[["ProjectedShare"]]

    st.bar_chart(chart_data)

    col1, col2 = st.columns(2)

    col1.metric(
        "Projected turnout voters",
        f"{projected_turnout_voters_total:,.0f}"
    )

    col2.metric(
        "Projected turnout rate",
        f"{projected_turnout_rate:.2%}"
    )


# ============================================================
# SELECTED WARDS
# ============================================================

elif page == "Selected Wards":

    st.header("2026 Selected Ward Projections")

    selected_wards = [
        "59500001",
        "59500036",
        "59500090"
    ]

    # Convert to strings for matching
    predictions.index = predictions.index.astype(str)

    available_selected = [
        ward for ward in selected_wards
        if ward in predictions.index
    ]

    for ward in available_selected:

        st.subheader(f"Ward {ward}")

        row = predictions.loc[ward]

        ward_results = pd.DataFrame({
            "Party": selected_parties,
            "ProjectedVoteShare": [
                row[party]
                for party in selected_parties
            ]
        })

        ward_results["ProjectedVoteShare"] *= 100

        st.dataframe(
            ward_results,
            use_container_width=True
        )

        leader = ward_results.loc[
            ward_results["ProjectedVoteShare"].idxmax()
        ]

        st.success(
            f"Projected leader: {leader['Party']} "
            f"({leader['ProjectedVoteShare']:.2f}%)"
        )

        st.metric(
            "Projected turnout",
            f"{row['Turnout']:.2%}"
        )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header("Model Evaluation")

    evaluation_rows = []

    for target, metrics in cv_results.items():

        evaluation_rows.append({
            "Target": target,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R²": metrics["R2"]
        })

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    st.dataframe(
        evaluation_df,
        use_container_width=True
    )

    st.subheader("Interpretation")

    st.write(
        """
        The models were evaluated using 5-fold cross-validation because
        only 53 geographically stable wards were available for modelling.

        Linear Regression was selected because the targets are continuous,
        the dataset is relatively small, and the model provides an
        interpretable relationship between historical ward characteristics
        and the subsequent election outcome.
        """
    )

    st.info(
        "The R² score should be interpreted separately for each target. "
        "Higher values indicate that the model explains more of the "
        "variation in that target."
    )


# ============================================================
# UNCERTAINTY & LIMITATIONS
# ============================================================

elif page == "Uncertainty & Limitations":

    st.header("Uncertainty & Limitations")

    st.subheader("Geographic Alignment")

    st.write(
        f"""
        The modelling dataset contains {len(stable_wards)} wards whose
        voting-district composition remained identical between 2016 and
        2021. These wards were used because they provide the strongest
        geographic comparability across election years.
        """
    )

    st.subheader("Small Modelling Dataset")

    st.write(
        """
        Only 53 stable wards were available for modelling. Therefore,
        5-fold cross-validation was used instead of relying on a single
        train/test split.
        """
    )

    st.subheader("Model Limitations")

    st.write(
        """
        Election outcomes can be affected by factors that are not contained
        in the historical election data, including campaigns, candidate
        changes, political events, voter behaviour and changes in ward
        boundaries.

        Therefore, the 2026 results are model projections rather than
        guaranteed election outcomes.
        """
    )

    st.subheader("Vote Share Treatment")

    st.write(
        """
        The selected parties are modelled individually. Their predicted
        shares are not renormalised to 100%, because other parties remain
        relevant and the residual share is retained rather than artificially
        redistributed.
        """
    )

    st.subheader("Coalition Interpretation")

    st.write(
        """
        The model indicates whether a single relevant party reaches a
        projected majority. It does not model negotiations or predict which
        parties would form a governing coalition.
        """
    )