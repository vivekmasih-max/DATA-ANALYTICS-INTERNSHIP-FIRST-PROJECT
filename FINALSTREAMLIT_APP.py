import streamlit as st

import pandas as pd

import joblib





# ============================================================

# STREAMLIT PAGE CONFIGURATION

# ============================================================



st.set_page_config(

    page_title="Nassau Candy Optimization System",

    page_icon="🍬",

    layout="wide"

)





# ============================================================

# LOAD DATA AND TRAINED PIPELINE

# ============================================================



@st.cache_data

def load_data():



    df = pd.read_csv(

        "C:/SEMESTER4/PROJECTWORK/CANDYDATA.csv"

    )



    return df





@st.cache_resource

def load_pipeline():



    pipeline = joblib.load(

        "gradient_boosting_pipeline.pkl"

    )



    return pipeline





df = load_data()

pipeline = load_pipeline()





# ============================================================

# FACTORY LIST

# ============================================================



factories = [

    "Wicked Choccy's",

    "Lot's O' Nuts",

    "Secret Factory",

    "The Other Factory",

    "Sugar Shack"

]





# ============================================================

# SESSION STATE

# ============================================================



if "simulation_results" not in st.session_state:

    st.session_state.simulation_results = None



if "selected_product" not in st.session_state:

    st.session_state.selected_product = None



if "selected_region" not in st.session_state:

    st.session_state.selected_region = None



if "selected_ship_mode" not in st.session_state:

    st.session_state.selected_ship_mode = None



if "current_factory" not in st.session_state:

    st.session_state.current_factory = None



if "best_factory" not in st.session_state:

    st.session_state.best_factory = None



if "base_scenario" not in st.session_state:

    st.session_state.base_scenario = None



if "current_pred_lead_time" not in st.session_state:

    st.session_state.current_pred_lead_time = None



if "current_profit_margin" not in st.session_state:

    st.session_state.current_profit_margin = None



if "current_distance" not in st.session_state:

    st.session_state.current_distance = None



if "speed_weight" not in st.session_state:

    st.session_state.speed_weight = 70



if "profit_weight" not in st.session_state:

    st.session_state.profit_weight = 30





# ============================================================

# MAIN SIMULATION FUNCTION

# ============================================================



def run_simulation(

    selected_product,

    selected_region,

    selected_ship_mode,

    speed_weight,

    profit_weight

):



    # --------------------------------------------------------

    # FILTER PRODUCT

    # --------------------------------------------------------



    product_df = df[

        df["Product Name"] == selected_product

    ]



    # --------------------------------------------------------

    # FILTER REGION

    # --------------------------------------------------------



    region_df = product_df[

        product_df["Region"] == selected_region

    ]



    # --------------------------------------------------------

    # GET BASE ROW

    # --------------------------------------------------------



    matching_rows = region_df[

        region_df["Ship Mode"] == selected_ship_mode

    ]



    if matching_rows.empty:



        raise ValueError(

            "No matching data found for the selected "

            "product, region and ship mode."

        )



    base_row = matching_rows.iloc[0].copy()



    current_factory = base_row["Factory"]



    # --------------------------------------------------------

    # BASE SCENARIO

    # --------------------------------------------------------



    base_scenario = pd.DataFrame(

        [base_row]

    )[

        [

            "Product Name",

            "Factory",

            "Region",

            "Ship Mode",

            "Distance_km(F TO C)",

            "Units",

            "Sales",

            "Cost",

            "Division"

        ]

    ]



    # --------------------------------------------------------

    # CURRENT FACTORY PREDICTION

    # --------------------------------------------------------



    current_pred_lead_time = pipeline.predict(

        base_scenario

    )[0]



    # --------------------------------------------------------

    # CURRENT FACTORY PROFIT MARGIN

    # --------------------------------------------------------



    current_factory_data = df[

        df["Factory"] == current_factory

    ]



    current_factory_data = current_factory_data[

        current_factory_data["Sales"] > 0

    ]



    if not current_factory_data.empty:



        current_profit_margin = (

            (

                current_factory_data["Sales"]

                - current_factory_data["Cost"]

            ).sum()

            /

            current_factory_data["Sales"].sum()

        ) * 100



    else:



        current_profit_margin = 0



    # --------------------------------------------------------

    # CURRENT DISTANCE

    # --------------------------------------------------------



    current_distance = base_scenario[

        "Distance_km(F TO C)"

    ].iloc[0]



    simulation_results = []



    # ========================================================

    # SIMULATE ALL FACTORIES

    # ========================================================



    for factory in factories:



        # ----------------------------------------------------

        # CREATE SCENARIO

        # ----------------------------------------------------



        test_scenario = base_scenario.copy()



        test_scenario["Factory"] = factory



        # ----------------------------------------------------

        # FIND FACTORY / PRODUCT DISTANCE

        # ----------------------------------------------------



        factory_data = df[

            (df["Product Name"] == selected_product)

            &

            (df["Factory"] == factory)

        ]



        if not factory_data.empty:



            test_scenario[

                "Distance_km(F TO C)"

            ] = (

                factory_data[

                    "Distance_km(F TO C)"

                ].median()

            )



        else:



            factory_all = df[

                df["Factory"] == factory

            ]



            if not factory_all.empty:



                test_scenario[

                    "Distance_km(F TO C)"

                ] = (

                    factory_all[

                        "Distance_km(F TO C)"

                    ].median()

                )



        # ----------------------------------------------------

        # PREDICT LEAD TIME

        # ----------------------------------------------------



        pred_lead_time = pipeline.predict(

            test_scenario

        )[0]



        # ----------------------------------------------------

        # HISTORICAL PROFIT MARGIN

        # ----------------------------------------------------



        candidate_data = df[

            df["Factory"] == factory

        ].copy()



        candidate_data = candidate_data[

            candidate_data["Sales"] > 0

        ]



        if not candidate_data.empty:



            candidate_profit_margin = (

                (

                    candidate_data["Sales"]

                    - candidate_data["Cost"]

                ).sum()

                /

                candidate_data["Sales"].sum()

            ) * 100



        else:



            candidate_profit_margin = (

                current_profit_margin

            )



        # ----------------------------------------------------

        # PROFIT IMPACT

        # ----------------------------------------------------



        profit_impact = (

            candidate_profit_margin

            - current_profit_margin

        )



        # ----------------------------------------------------

        # PROFIT STATUS

        # ----------------------------------------------------



        if profit_impact > 1:



            profit_status = "🟢 Positive"



        elif profit_impact >= -1:



            profit_status = "🟡 Neutral"



        else:



            profit_status = "🔴 Negative"



        # ----------------------------------------------------

        # DISTANCE

        # ----------------------------------------------------



        candidate_distance = test_scenario[

            "Distance_km(F TO C)"

        ].iloc[0]



        if current_distance != 0:



            distance_change_pct = (

                (

                    candidate_distance

                    - current_distance

                )

                /

                current_distance

            ) * 100



        else:



            distance_change_pct = 0



        # ----------------------------------------------------

        # OBSERVATION COUNT

        # ----------------------------------------------------



        observation_count = len(

            candidate_data

        )



        # ----------------------------------------------------

        # LEAD TIME IMPROVEMENT

        # ----------------------------------------------------



        if current_pred_lead_time != 0:



            lead_time_improvement = (

                (

                    current_pred_lead_time

                    - pred_lead_time

                )

                /

                current_pred_lead_time

            ) * 100



        else:



            lead_time_improvement = 0



        # ====================================================

        # RISK LOGIC

        # ====================================================



        risk_points = 0



        # ----------------------------------------------------

        # 1. POOR HISTORICAL PERFORMANCE

        # ----------------------------------------------------



        if (

            candidate_profit_margin

            < current_profit_margin - 5

        ):



            risk_points += 2



        elif (

            candidate_profit_margin

            < current_profit_margin

        ):



            risk_points += 1



        # ----------------------------------------------------

        # 2. FEW OBSERVATIONS

        # ----------------------------------------------------



        if observation_count < 50:



            risk_points += 2



        elif observation_count < 100:



            risk_points += 1



        # ----------------------------------------------------

        # 3. SIGNIFICANT PROFIT DETERIORATION

        # ----------------------------------------------------



        if profit_impact < -5:



            risk_points += 2



        elif profit_impact < -2:



            risk_points += 1



        # ----------------------------------------------------

        # 4. LEAD TIME GETS WORSE

        # ----------------------------------------------------



        if lead_time_improvement < -10:



            risk_points += 2



        elif lead_time_improvement < 0:



            risk_points += 1



        # ----------------------------------------------------

        # 5. DISTANCE INCREASE

        #

        # NEW DISTANCE RISK FACTOR

        #

        # No distance weight is used.

        # Distance affects recommendation through risk.

        # ----------------------------------------------------



        if distance_change_pct > 20:



            risk_points += 2



        elif distance_change_pct > 10:



            risk_points += 1



        # ----------------------------------------------------

        # FINAL RISK

        # ----------------------------------------------------



        if risk_points >= 4:



            risk = "HIGH"



        elif risk_points >= 2:



            risk = "MEDIUM"



        else:



            risk = "LOW"



        # ----------------------------------------------------

        # STORE RESULTS

        # ----------------------------------------------------



        simulation_results.append({



            "Factory": factory,



            "Predicted Lead Time":

                pred_lead_time,



            "Lead Time Improvement (%)":

                lead_time_improvement,



            "Profit Impact (%)":

                profit_impact,



            "Profit Impact":

                profit_status,



            "Distance Change (%)":

                distance_change_pct,



            "Distance (km)":

                candidate_distance,



            "Observations":

                observation_count,



            "Risk":

                risk

        })



    # ========================================================

    # RESULTS DATAFRAME

    # ========================================================



    results_df = pd.DataFrame(

        simulation_results

    )



    # ========================================================

    # OPTIMIZATION WEIGHTS

    # ========================================================



    speed_weight_decimal = (

        speed_weight / 100

    )



    profit_weight_decimal = (

        profit_weight / 100

    )



    # ========================================================
    # OPTIMIZATION SCORE
    # ========================================================
    # Distance has its own optimization weight in addition to
    # its existing effect on risk and recommendation rules.
    # Shorter distance improves the score; longer distance
    # reduces the score.
    # ========================================================

    DISTANCE_WEIGHT = 0.10

    results_df["Optimization Score"] = (
        results_df["Lead Time Improvement (%)"]
        * speed_weight_decimal
        + results_df["Profit Impact (%)"]
        * profit_weight_decimal
        - results_df["Distance Change (%)"]
        * DISTANCE_WEIGHT
        - results_df["Risk"].map({
            "LOW": 0,
            "MEDIUM": 5,
            "HIGH": 10
        })
        * 0.20
    )



    # ========================================================

    # BEST FACTORY

    # ========================================================



    candidate_factories = results_df[

        results_df["Factory"] != current_factory

    ]



    if not candidate_factories.empty:



        best_factory = candidate_factories.loc[

            candidate_factories[

                "Optimization Score"

            ].idxmax(),

            "Factory"

        ]



    else:



        best_factory = current_factory



    # ========================================================

    # RECOMMENDATION LOGIC

    #

    # DISTANCE IS NOW CONSIDERED

    #

    # No distance weight.

    # No distance KPI.

    # ========================================================



    recommendations = []



    for _, row in results_df.iterrows():



        factory = row["Factory"]



        lead_improvement = (

            row["Lead Time Improvement (%)"]

        )



        profit_impact = (

            row["Profit Impact (%)"]

        )



        distance_change = (

            row["Distance Change (%)"]

        )



        risk = row["Risk"]



        if factory == current_factory:



            recommendation = "Current"



        # ----------------------------------------------------

        # RECOMMENDED

        #

        # Distance must not increase by more than 10%.

        # ----------------------------------------------------



        elif (

            factory == best_factory

            and lead_improvement > 0

            and profit_impact >= 0

            and distance_change <= 10

            and risk != "HIGH"

        ):



            recommendation = "⭐ Recommended"



        # ----------------------------------------------------

        # GOOD

        #

        # Good lead-time and profit performance with

        # acceptable distance.

        # ----------------------------------------------------



        elif (

            lead_improvement > 0

            and profit_impact >= 0

            and distance_change <= 10

            and risk == "LOW"

        ):



            recommendation = "Good"



        # ----------------------------------------------------

        # MODERATE

        #

        # Allows small profit deterioration, but distance

        # must still remain within acceptable range.

        # ----------------------------------------------------



        elif (

            lead_improvement > 0

            and profit_impact >= -2

            and distance_change <= 20

            and risk != "HIGH"

        ):



            recommendation = "Moderate"



        else:



            recommendation = "Avoid"



        recommendations.append(

            recommendation

        )



    results_df[

        "Recommendation"

    ] = recommendations



    # ========================================================

    # FORMATTING

    # ========================================================



    results_df[

        "Predicted Lead Time Display"

    ] = results_df[

        "Predicted Lead Time"

    ].apply(

        lambda x: f"{x:.2f}"

    )



    results_df[

        "Improvement"

    ] = results_df[

        "Lead Time Improvement (%)"

    ].apply(

        lambda x: f"{x:+.1f}%"

    )



    results_df[

        "Profit Impact Value"

    ] = results_df[

        "Profit Impact (%)"

    ].apply(

        lambda x: f"{x:+.1f}%"

    )



    results_df[

        "Distance Change"

    ] = results_df[

        "Distance Change (%)"

    ].apply(

        lambda x: f"{x:+.1f}%"

    )



    results_df[

        "Distance"

    ] = results_df[

        "Distance (km)"

    ].apply(

        lambda x: f"{x:,.2f} km"

    )



    # ========================================================

    # DEFAULT INTERNAL ORDER

    # ========================================================



    results_df = results_df.sort_values(

        by="Optimization Score",

        ascending=False

    ).reset_index(drop=True)



    return (

        results_df,

        current_factory,

        current_pred_lead_time,

        current_profit_margin,

        current_distance,

        best_factory,

        base_scenario

    )





# ============================================================

# SIDEBAR NAVIGATION

# ============================================================



st.sidebar.title("🍬 Nassau Candy")



st.sidebar.markdown(

    "### Optimization System"

)



page = st.sidebar.radio(

    "Menu",

    [

        "🏠 Home",

        "🏭 Factory Optimization",

        "🔄 What-If Analysis",

        "💡 Recommendations",

        "⚠️ Risk & Impact"

    ]

)



st.sidebar.markdown("---")



st.sidebar.caption(

    "Factory Reallocation & Shipping "

    "Optimization Recommendation System"

)





# ============================================================

# HOME PAGE

# ============================================================



if page == "🏠 Home":



    st.title(

        "🍬 Nassau Candy Optimization System"

    )



    st.subheader(

        "Factory Reallocation & Shipping "

        "Optimization Recommendation System"

    )



    st.markdown("---")



    st.markdown(

        """

        ### Welcome



        This application helps analyze whether a product could

        perform better when assigned to a different factory.



        The system combines:



        - Machine Learning based lead-time prediction

        - Historical factory profitability

        - Factory simulation

        - What-if analysis

        - Optimization priority control

        - Recommendation logic

        - Risk assessment

        """

    )



    st.markdown("---")



    st.subheader(

        "How the system works"

    )



    col1, col2, col3, col4 = st.columns(4)



    with col1:



        st.markdown(

            "### 🏭\nFactory Optimization"

        )



        st.caption(

            "Compare predicted performance "

            "across factories."

        )



    with col2:



        st.markdown(

            "### 🔄\nWhat-If Analysis"

        )



        st.caption(

            "Compare current and alternative "

            "factory assignments."

        )



    with col3:



        st.markdown(

            "### 💡\nRecommendations"

        )



        st.caption(

            "Identify potential factory "

            "reassignment opportunities."

        )



    with col4:



        st.markdown(

            "### ⚠️\nRisk & Impact"

        )



        st.caption(

            "Understand profit and "

            "operational risks."

        )



    st.markdown("---")



    st.info(

        "Use the menu on the left to move directly "

        "to any section of the application."

    )





# ============================================================

# FACTORY OPTIMIZATION PAGE

# ============================================================



elif page == "🏭 Factory Optimization":



    st.title(

        "🏭 Factory Optimization Simulator"

    )



    st.write(

        "Select a product, region and shipping mode "

        "to simulate its performance across factories."

    )



    st.markdown("---")



    # --------------------------------------------------------

    # PRODUCT

    # --------------------------------------------------------



    products = sorted(

        df["Product Name"]

        .dropna()

        .unique()

    )



    selected_product = st.selectbox(

        "Select Product",

        products

    )



    # --------------------------------------------------------

    # REGION

    # --------------------------------------------------------



    product_df = df[

        df["Product Name"]

        == selected_product

    ]



    regions = sorted(

        product_df["Region"]

        .dropna()

        .unique()

    )



    selected_region = st.selectbox(

        "Select Region",

        regions

    )



    # --------------------------------------------------------

    # SHIP MODE

    # --------------------------------------------------------



    region_df = product_df[

        product_df["Region"]

        == selected_region

    ]



    ship_modes = sorted(

        region_df["Ship Mode"]

        .dropna()

        .unique()

    )



    selected_ship_mode = st.selectbox(

        "Select Ship Mode",

        ship_modes

    )



    st.markdown("---")



    # ========================================================

    # OPTIMIZATION PRIORITY

    # ========================================================



    st.subheader(

        "⚙️ Optimization Priority"

    )



    st.write(

        "Choose whether the optimization should prioritize "

        "faster delivery or higher profitability."

    )



    # --------------------------------------------------------

    # SAME SPEED-PROFIT SLIDER

    # --------------------------------------------------------



    speed_weight = st.slider(

        "Speed vs Profit",

        min_value=0,

        max_value=100,

        value=70,

        step=10,

        format="%d%%"

    )



    profit_weight = 100 - speed_weight



    col1, col2 = st.columns(2)



    with col1:



        st.metric(

            "⚡ Speed Priority",

            f"{speed_weight}%"

        )



    with col2:



        st.metric(

            "💰 Profit Priority",

            f"{profit_weight}%"

        )



    if speed_weight > profit_weight:



        st.info(

            f"Current strategy prioritizes **speed** "

            f"({speed_weight}%) over profit "

            f"({profit_weight}%)."

        )



    elif profit_weight > speed_weight:



        st.info(

            f"Current strategy prioritizes **profit** "

            f"({profit_weight}%) over speed "

            f"({speed_weight}%)."

        )



    else:



        st.info(

            "Current strategy gives **equal priority** "

            "to speed and profit (50% / 50%)."

        )



    st.markdown("---")



    # ========================================================

    # RUN SIMULATION

    # ========================================================



    if st.button(

        "🚀 Run Factory Simulation",

        type="primary"

    ):



        try:



            (

                results_df,

                current_factory,

                current_pred_lead_time,

                current_profit_margin,

                current_distance,

                best_factory,

                base_scenario

            ) = run_simulation(

                selected_product,

                selected_region,

                selected_ship_mode,

                speed_weight,

                profit_weight

            )



            st.session_state.simulation_results = (

                results_df

            )



            st.session_state.selected_product = (

                selected_product

            )



            st.session_state.selected_region = (

                selected_region

            )



            st.session_state.selected_ship_mode = (

                selected_ship_mode

            )



            st.session_state.current_factory = (

                current_factory

            )



            st.session_state.current_pred_lead_time = (

                current_pred_lead_time

            )



            st.session_state.current_profit_margin = (

                current_profit_margin

            )



            st.session_state.current_distance = (

                current_distance

            )



            st.session_state.best_factory = (

                best_factory

            )



            st.session_state.base_scenario = (

                base_scenario

            )



            st.session_state.speed_weight = (

                speed_weight

            )



            st.session_state.profit_weight = (

                profit_weight

            )



            st.success(

                "Factory simulation completed successfully."

            )



        except Exception as e:



            st.error(

                f"Simulation failed: {e}"

            )



    # ========================================================

    # DISPLAY RESULTS

    # ========================================================



    if (

        st.session_state.simulation_results

        is not None

    ):



        results_df = (

            st.session_state.simulation_results

        )



        st.markdown("---")



        st.subheader(

            "Current Factory"

        )



        col1, col2, col3, col4 = st.columns(4)



        with col1:



            st.metric(

                "Product",

                st.session_state.selected_product

            )



        with col2:



            st.metric(

                "Current Factory",

                st.session_state.current_factory

            )



        with col3:



            st.metric(

                "Predicted Lead Time",

                f"{st.session_state.current_pred_lead_time:.2f}"

            )



        with col4:



            st.metric(

                "Profit Margin",

                f"{st.session_state.current_profit_margin:.2f}%"

            )



        st.markdown("---")



        st.subheader(

            "Active Optimization Strategy"

        )



        col1, col2, col3 = st.columns(3)



        with col1:



            st.metric(

                "⚡ Speed Weight",

                f"{st.session_state.speed_weight}%"

            )



        with col2:



            st.metric(

                "💰 Profit Weight",

                f"{st.session_state.profit_weight}%"

            )



        with col3:



            st.metric(

                "⚠️ Risk Penalty",

                "20%"

            )



        st.markdown("---")



        # ====================================================

        # FACTORY COMPARISON

        # ====================================================



        st.subheader(

            "Factory Performance Comparison"

        )



        risk_order = {

            "LOW": 1,

            "MEDIUM": 2,

            "HIGH": 3

        }



        display_table = results_df[

            [

                "Factory",

                "Predicted Lead Time Display",

                "Improvement",

                "Distance",

                "Distance Change",

                "Profit Impact Value",

                "Profit Impact",

                "Risk",

                "Optimization Score",

                "Recommendation"

            ]

        ].copy()



        display_table["Risk Priority"] = (

            display_table["Risk"].map(risk_order)

        )



        display_table = display_table.sort_values(

            by=[

                "Risk Priority",

                "Optimization Score"

            ],

            ascending=[

                True,

                False

            ]

        ).reset_index(drop=True)



        display_table = display_table.drop(

            columns=["Risk Priority"]

        )



        st.dataframe(

            display_table,

            use_container_width=True,

            hide_index=True

        )



        st.markdown("---")



        st.success(

            f"Optimization Recommendation: "

            f"{st.session_state.best_factory}"

        )



        st.caption(

            f"The recommendation uses "

            f"{st.session_state.speed_weight}% speed weight "

            f"and "

            f"{st.session_state.profit_weight}% profit weight. "

            f"Distance is considered through the existing "

            f"risk and recommendation logic, but has no "

            f"separate optimization weight."

        )





# ============================================================

# WHAT-IF ANALYSIS PAGE

# ============================================================



elif page == "🔄 What-If Analysis":



    st.title(

        "🔄 What-If Scenario Analysis"

    )



    if (

        st.session_state.simulation_results

        is None

    ):



        st.info(

            "Please go to Factory Optimization "

            "and run a simulation first."

        )



    else:



        results_df = (

            st.session_state.simulation_results

        )



        current_factory = (

            st.session_state.current_factory

        )



        current_row = results_df[

            results_df["Factory"]

            == current_factory

        ].iloc[0]



        st.subheader(

            "Current Assignment"

        )



        col1, col2, col3, col4 = st.columns(4)



        with col1:



            st.metric(

                "Current Factory",

                current_factory

            )



        with col2:



            st.metric(

                "Lead Time",

                f"{current_row['Predicted Lead Time']:.2f}"

            )



        with col3:



            st.metric(

                "Distance",

                current_row["Distance"]

            )



        with col4:



            st.metric(

                "Profit Impact",

                current_row["Profit Impact Value"]

            )



        st.markdown("---")



        st.subheader(

            "Current Optimization Priority"

        )



        col1, col2 = st.columns(2)



        with col1:



            st.metric(

                "⚡ Speed",

                f"{st.session_state.speed_weight}%"

            )



        with col2:



            st.metric(

                "💰 Profit",

                f"{st.session_state.profit_weight}%"

            )



        st.markdown("---")



        st.subheader(

            "Compare Alternative Factory"

        )



        alternative_factories = [

            f for f in factories

            if f != current_factory

        ]



        selected_factory = st.selectbox(

            "Select Alternative Factory",

            alternative_factories

        )



        selected_row = results_df[

            results_df["Factory"]

            == selected_factory

        ].iloc[0]



        st.markdown("---")



        col1, col2 = st.columns(2)



        with col1:



            st.markdown(

                "### Current Assignment"

            )



            st.write(

                f"**Factory:** {current_factory}"

            )



            st.write(

                f"**Lead Time:** "

                f"{current_row['Predicted Lead Time']:.2f}"

            )



            st.write(

                f"**Distance:** "

                f"{current_row['Distance']}"

            )



            st.write(

                f"**Profit Impact:** "

                f"{current_row['Profit Impact Value']}"

            )



        with col2:



            st.markdown(

                "### Alternative Assignment"

            )



            st.write(

                f"**Factory:** {selected_factory}"

            )



            st.write(

                f"**Lead Time:** "

                f"{selected_row['Predicted Lead Time']:.2f}"

            )



            st.write(

                f"**Distance:** "

                f"{selected_row['Distance']}"

            )



            st.write(

                f"**Profit Impact:** "

                f"{selected_row['Profit Impact Value']}"

            )



        st.markdown("---")



        st.subheader(

            "Scenario Impact"

        )



        col1, col2, col3 = st.columns(3)



        with col1:



            st.metric(

                "Lead Time Improvement",

                selected_row["Improvement"]

            )



        with col2:



            st.metric(

                "Distance Change",

                selected_row["Distance Change"]

            )



        with col3:



            st.metric(

                "Risk",

                selected_row["Risk"]

            )



        st.info(

            "Distance is shown for operational analysis "

            "and is also considered in the recommendation "

            "and risk logic."

        )





# ============================================================

# RECOMMENDATIONS PAGE

# ============================================================



elif page == "💡 Recommendations":



    st.title(

        "💡 Recommendation Dashboard"

    )



    if (

        st.session_state.simulation_results

        is None

    ):



        st.info(

            "Please run a factory simulation first."

        )



    else:



        results_df = (

            st.session_state.simulation_results

        )



        current_factory = (

            st.session_state.current_factory

        )



        best_factory = (

            st.session_state.best_factory

        )



        st.subheader(

            "Optimization Recommendation"

        )



        st.success(

            f"Recommended Factory: {best_factory}"

        )



        st.write(

            f"Current Factory: **{current_factory}**"

        )



        st.markdown("---")



        st.subheader(

            "Optimization Priority Used"

        )



        col1, col2 = st.columns(2)



        with col1:



            st.metric(

                "⚡ Speed",

                f"{st.session_state.speed_weight}%"

            )



        with col2:



            st.metric(

                "💰 Profit",

                f"{st.session_state.profit_weight}%"

            )



        st.markdown("---")



        # ====================================================

        # RECOMMENDATION TABLE

        # ====================================================



        st.subheader(

            "Recommendation Ranking"

        )



        recommendation_order = {

            "⭐ Recommended": 1,

            "Good": 2,

            "Moderate": 3,

            "Avoid": 4,

            "Current": 5

        }



        recommendation_df = results_df[

            [

                "Factory",

                "Predicted Lead Time Display",

                "Improvement",

                "Distance",

                "Distance Change",

                "Profit Impact Value",

                "Profit Impact",

                "Risk",

                "Optimization Score",

                "Recommendation"

            ]

        ].copy()



        recommendation_df[

            "Recommendation Priority"

        ] = (

            recommendation_df[

                "Recommendation"

            ].map(recommendation_order)

        )



        recommendation_df = (

            recommendation_df.sort_values(

                by=[

                    "Recommendation Priority",
                    
                    "Optimization Score"

                   

                ],

                ascending=[

                    True,

                    False

                ]

            )

            .reset_index(drop=True)

        )



        # ----------------------------------------------------

        # ADD RANK AFTER SORTING

        # ----------------------------------------------------



        recommendation_df.insert(

            0,

            "Rank",

            range(

                1,

                len(recommendation_df) + 1

            )

        )



        # ----------------------------------------------------

        # REMOVE HELPER COLUMN

        # ----------------------------------------------------



        recommendation_df = (

            recommendation_df.drop(

                columns=[

                    "Recommendation Priority"

                ]

            )

        )



        st.dataframe(

            recommendation_df,

            use_container_width=True,

            hide_index=True

        )



        st.markdown("---")



        # ====================================================

        # POTENTIAL REASSIGNMENTS

        # ====================================================



        st.subheader(

            "Potential Reassignments"

        )



        recommended_rows = results_df[

            results_df["Recommendation"]

            == "⭐ Recommended"

        ]



        if not recommended_rows.empty:



            for _, row in recommended_rows.iterrows():



                st.success(

                    f"**{row['Factory']}** → "

                    f"Lead Time Improvement: "

                    f"{row['Improvement']} | "

                    f"Profit Impact: "

                    f"{row['Profit Impact Value']} | "

                    f"Risk: {row['Risk']}"

                )



        else:



            st.warning(

                "No factory currently satisfies all "

                "recommendation conditions."

            )



        st.markdown("---")



        # ====================================================

        # RECOMMENDATION KPIs

        # ====================================================



        st.subheader("Recommendation KPIs")



        # ============================================================
        # RECOMMENDATION COVERAGE
        # ============================================================

        VALID_LABELS = ["⭐ Recommended", "Good", "Moderate"]

        all_products = df["Product Name"].dropna().unique()

        eligible_products = set()
        covered_products = set()

        for product in all_products:

            product_data = df[df["Product Name"] == product]

            combinations = (
                product_data[["Region", "Ship Mode"]]
                .dropna()
                .drop_duplicates()
            )

            for _, combination in combinations.iterrows():

                try:
                    results, *_ = run_simulation(
                        product,
                        combination["Region"],
                        combination["Ship Mode"],
                        st.session_state.speed_weight,
                        st.session_state.profit_weight
                    )

                    # A product is eligible if the simulation returns results
                    if results is None or results.empty:
                        continue

                    eligible_products.add(product)

                    # Check for at least one valid alternative factory
                    valid_recommendations = results[
                        results["Recommendation"].isin(VALID_LABELS)
                    ]

                    if not valid_recommendations.empty:
                        covered_products.add(product)
                        break

                except (ValueError, IndexError, KeyError):
                    continue

        total_eligible_products = len(eligible_products)
        products_with_recommendations = len(covered_products)

        recommendation_coverage = (
            products_with_recommendations / total_eligible_products * 100
            if total_eligible_products > 0
            else 0
        )

       



        # ----------------------------------------------------

        # 2. SCENARIO CONFIDENCE SCORE

        # ----------------------------------------------------



        score_min = results_df[

            "Optimization Score"

        ].min()



        score_max = results_df[

            "Optimization Score"

        ].max()



        if score_max > score_min:



            results_df["Score Strength"] = (

                (

                    results_df[

                        "Optimization Score"

                    ] - score_min

                )

                /

                (

                    score_max - score_min

                )

            ) * 100



        else:



            results_df["Score Strength"] = 100



        recommendation_confidence = {

            "⭐ Recommended": 100,

            "Good": 80,

            "Moderate": 60,

            "Avoid": 30,

            "Current": 50

        }



        results_df[

            "Recommendation Confidence"

        ] = results_df[

            "Recommendation"

        ].map(

            recommendation_confidence

        ).fillna(50)



        risk_confidence = {

            "LOW": 100,

            "MEDIUM": 70,

            "HIGH": 40

        }



        results_df[

            "Risk Confidence"

        ] = results_df[

            "Risk"

        ].str.upper().map(

            risk_confidence

        ).fillna(50)



        # ----------------------------------------------------

        # Calculate scenario confidence

        # ----------------------------------------------------



        results_df[

            "Scenario Confidence"

        ] = (



            results_df[

                "Score Strength"

            ] * 0.40



            +



            results_df[

                "Recommendation Confidence"

            ] * 0.35



            +



            results_df[

                "Risk Confidence"

            ] * 0.25



        )



        recommended_factory_row = results_df[

            results_df["Factory"]

            == best_factory

        ]



        if not recommended_factory_row.empty:



            scenario_confidence = (

                recommended_factory_row[

                    "Scenario Confidence"

                ].iloc[0]

            )



        else:



            scenario_confidence = 0



        # ----------------------------------------------------

        # DISPLAY KPI CARDS

        # ----------------------------------------------------



        kpi1, kpi2 = st.columns(2)



        with kpi1:



            st.metric(

                "🎯 Scenario Confidence Score",

                f"{scenario_confidence:.0f}%"

            )



            st.caption(

                "Decision-support confidence based on "

                "optimization score, recommendation outcome "

                "and risk."

            )



        with kpi2:



            st.metric(

                "📊 Recommendation Coverage",

                f"{recommendation_coverage:.0f}%"

            )



            st.caption(
    f"{products_with_recommendations} of "
    f"{total_eligible_products} eligible products have "
    "at least one valid recommendation."
)



        st.markdown("---")



        st.info(

            "Factories are ranked here according to the "

            "recommendation outcome first, and Optimization "

            "Score second. The Optimization Score considers "

            "lead-time improvement, profit impact and risk "

            "penalty. Distance is considered through the "

            "risk and recommendation logic, but has no "

            "separate optimization weight."

        )





# ============================================================

# RISK & IMPACT PAGE

# ============================================================



elif page == "⚠️ Risk & Impact":



    st.title(

        "⚠️ Risk & Impact Panel"

    )



    if (

        st.session_state.simulation_results

        is None

    ):



        st.info(

            "Please run a factory simulation first."

        )



    else:



        results_df = (

            st.session_state.simulation_results

        )



        st.subheader(

            "Risk Overview"

        )



        high_risk = len(

            results_df[

                results_df["Risk"] == "HIGH"

            ]

        )



        medium_risk = len(

            results_df[

                results_df["Risk"] == "MEDIUM"

            ]

        )



        low_risk = len(

            results_df[

                results_df["Risk"] == "LOW"

            ]

        )



        col1, col2, col3 = st.columns(3)



        with col1:



            st.metric(

                "🔴 High Risk",

                high_risk

            )



        with col2:



            st.metric(

                "🟡 Medium Risk",

                medium_risk

            )



        with col3:



            st.metric(

                "🟢 Low Risk",

                low_risk

            )



        st.markdown("---")



        st.subheader(

            "Current Optimization Priority"

        )



        col1, col2 = st.columns(2)



        with col1:



            st.metric(

                "⚡ Speed",

                f"{st.session_state.speed_weight}%"

            )



        with col2:



            st.metric(

                "💰 Profit",

                f"{st.session_state.profit_weight}%"

            )



        st.markdown("---")



        st.subheader(

            "Profit Impact Alerts"

        )



        negative_profit = results_df[

            results_df["Profit Impact (%)"] < 0

        ]



        if not negative_profit.empty:



            for _, row in negative_profit.iterrows():



                st.error(

                    f"{row['Factory']}: "

                    f"Profit impact "

                    f"{row['Profit Impact Value']}"

                )



        else:



            st.success(

                "No negative profit impact detected."

            )



        st.markdown("---")



        st.subheader(

            "High-Risk Factory Reassignments"

        )



        high_risk_df = results_df[

            results_df["Risk"] == "HIGH"

        ]



        if not high_risk_df.empty:



            st.dataframe(

                high_risk_df[

                    [

                        "Factory",

                        "Predicted Lead Time Display",

                        "Improvement",

                        "Distance",

                        "Profit Impact Value",

                        "Risk",

                        "Recommendation"

                    ]

                ],

                use_container_width=True,

                hide_index=True

            )



        else:



            st.success(

                "No high-risk factory reassignment "

                "identified."

            )



        st.markdown("---")



        st.subheader(

            "Risk Logic"

        )



        st.markdown(

            """

            **LOW**

            - Few or no significant risk factors



            **MEDIUM**

            - Some risk factors are present



            **HIGH**

            - Multiple significant risk factors



            """

        )



        st.info(

            "Distance change is displayed as an operational "

            "metric and is also considered in the risk and "

            "recommendation logic. It does not have a separate "

            "optimization weight."

        )