import streamlit as st
import numpy as np
import pandas as pd
import pulp  # Import the MILP solver library
import math

# Set page configuration
st.set_page_config(
    page_title="Integrated DFS-AHP-QFD-MILP Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
:root {
    --primary: #1f77b4;
    --secondary: #2ca02c;
    --accent: #ff6b6b;
    --background: #f8f9fa;
    --card-bg: #ffffff;
    --text: #262730;
    --border-radius: 12px;
    --shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}
.main-header {
    font-size: 2.8rem;
    color: var(--primary);
    text-align: center;
    padding: 1.5rem 0;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-header {
    font-size: 1.6rem;
    color: var(--primary);
    border-left: 5px solid var(--secondary);
    padding-left: 1rem;
    margin: 2rem 0 1.5rem 0;
    font-weight: 600;
}
.panel {
    background-color: var(--card-bg);
    border-radius: var(--border-radius);
    padding: 1.8rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    border: 1px solid #e0e0e0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.panel:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}
.stButton>button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
    width: 100%;
    margin-top: 1rem;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    background: linear-gradient(135deg, var(--secondary), var(--primary));
}
.instruction-box {
    background-color: #f0f7ff;
    border-left: 4px solid var(--primary);
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin: 1rem 0;
    font-size: 0.95rem;
}
.success-box {
    background-color: #f0fff4;
    border-left: 4px solid var(--secondary);
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fffaf0;
    border-left: 4px solid #ffb347;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin: 1rem 0;
}
.error-box {
    background-color: #fff5f5;
    border-left: 4px solid #ff6b6b;
    padding: 1rem 1.5rem;
    border-radius: 4px;
    margin: 1rem 0;
}
.logo-container {
    text-align: center;
    margin-bottom: 1.5rem;
}
.logo {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ==================== DFS DEFINITIONS AND OPERATIONS ====================

# DFS Linguistic Scale (from Table 2 in the paper)
dfs_linguistic_scale = {
    'EEI': {'O': (0.50, 0.50), 'P': (0.50, 0.50)},  # Exactly Equal Importance
    'SMI': {'O': (0.55, 0.45), 'P': (0.45, 0.55)},  # Slightly More Important
    'WMI': {'O': (0.60, 0.40), 'P': (0.40, 0.60)},  # Weakly More Important
    'MI': {'O': (0.65, 0.35), 'P': (0.35, 0.65)},   # More Important
    'SMM': {'O': (0.70, 0.30), 'P': (0.30, 0.70)},  # Strongly More Important
    'VSI': {'O': (0.75, 0.25), 'P': (0.25, 0.75)},  # Very Strongly More Important
    'AMI': {'O': (0.80, 0.20), 'P': (0.20, 0.80)},  # Absolutely More Important
    'PMI': {'O': (0.85, 0.15), 'P': (0.15, 0.85)},  # Perfectly More Important
    'EMI': {'O': (0.90, 0.10), 'P': (0.10, 0.90)},  # Exactly More Important
}

linguistic_options = list(dfs_linguistic_scale.keys())

# DFS Operations (from Definitions 2 and 3)
def dfs_addition(dfs1, dfs2):
    """DFS addition operation"""
    O_a1, O_b1 = dfs1['O']
    O_a2, O_b2 = dfs2['O']
    P_c1, P_d1 = dfs1['P']
    P_c2, P_d2 = dfs2['P']
    
    O_a = O_a1 + O_a2 - 2 * O_a1 * O_a2
    if O_a1 * O_a2 != 1:
        O_a /= (1 - O_a1 * O_a2)
    
    O_b = (O_b1 * O_b2) / (O_b1 + O_b2 - O_b1 * O_b2) if (O_b1 + O_b2 - O_b1 * O_b2) != 0 else 0
    
    P_c = P_c1 + P_c2 - P_c1 * P_c2
    P_d = P_d1 * P_d2
    
    return {'O': (O_a, O_b), 'P': (P_c, P_d)}

def dfs_multiplication(dfs1, dfs2):
    """DFS multiplication operation"""
    O_a1, O_b1 = dfs1['O']
    O_a2, O_b2 = dfs2['O']
    P_c1, P_d1 = dfs1['P']
    P_c2, P_d2 = dfs2['P']
    
    O_a = O_a1 * O_a2
    O_b = O_b1 + O_b2 - O_b1 * O_b2
    
    P_c = (P_c1 * P_c2) / (P_c1 + P_c2 - P_c1 * P_c2) if (P_c1 + P_c2 - P_c1 * P_c2) != 0 else 0
    P_d = P_d1 + P_d2 - 2 * P_d1 * P_d2
    if P_d1 * P_d2 != 1:
        P_d /= (1 - P_d1 * P_d2)
    
    return {'O': (O_a, O_b), 'P': (P_c, P_d)}

def dfs_dwgm(dfs_list, weights):
    """Decomposed Weighted Geometric Mean (DWGM) operator"""
    if len(dfs_list) != len(weights):
        raise ValueError("DFS list and weights must have the same length")
    
    n = len(dfs_list)
    
    # Initialize products
    O_a_product = 1.0
    O_b_product = 1.0
    P_c_product = 1.0
    P_d_product = 1.0
    
    O_b_complement_product = 1.0
    P_c_complement_sum = 0.0
    
    for i, dfs in enumerate(dfs_list):
        O_a, O_b = dfs['O']
        P_c, P_d = dfs['P']
        w = weights[i]
        
        O_a_product *= (O_a ** w)
        O_b_complement_product *= ((1 - O_b) ** w)
        P_c_product *= (P_c ** w)
        P_d_product *= (P_d ** w)
        
        P_c_complement_sum += (w * (1 - P_c)) / (P_c ** (n-1)) if P_c != 0 else 0
    
    O_b_new = 1 - O_b_complement_product
    
    P_c_denominator = P_c_complement_sum + P_c_product
    P_c_new = P_c_product / P_c_denominator if P_c_denominator != 0 else 0
    
    P_d_numerator = sum(weights[i] * dfs_list[i]['P'][1] for i in range(n))
    P_d_denominator = 1 + sum((weights[i] * dfs_list[i]['P'][1] - dfs_list[i]['P'][1]) / n for i in range(n))
    P_d_new = P_d_numerator / P_d_denominator if P_d_denominator != 0 else 0
    
    return {'O': (O_a_product, O_b_new), 'P': (P_c_new, P_d_new)}

def dfs_consistency_index(dfs):
    """Calculate Consistency Index (CI) for DFS number"""
    O_a, O_b = dfs['O']
    P_c, P_d = dfs['P']
    
    numerator = ((O_a - P_d) ** 2 + (O_b - P_c) ** 2 + 
                 (1 - O_a - O_b) ** 2 + (1 - P_c - P_d) ** 2)
    
    CI = 1 - math.sqrt(numerator / 2)
    return max(0, min(1, CI))  # Ensure between 0 and 1

def dfs_defuzzification(dfs):
    """Defuzzify DFS number to crisp value (Equation 12)"""
    O_a, O_b = dfs['O']
    P_c, P_d = dfs['P']
    
    numerator = (O_a + O_b - P_c + P_d) * dfs_consistency_index(dfs)
    denominator = 2 * 0.9  # k = 0.9 as per the paper
    
    return numerator / denominator if denominator != 0 else 0

# ==================== ML-BASED EXPERT WEIGHTING ====================

def ml_expert_weighting(expert_data, cov_matrix=None):
    """
    ML-based expert weighting using PCA approach
    expert_data: numpy array of shape (n_experts, n_features)
    """
    # If covariance matrix is not provided, compute it from data
    if cov_matrix is None:
        # Normalize the data
        min_vals = np.min(expert_data, axis=0)
        max_vals = np.max(expert_data, axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero if a feature is constant
        range_vals[range_vals == 0] = 1
        expert_data_normalized = (expert_data - max_vals) / range_vals
        cov_matrix = np.cov(expert_data_normalized.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Select eigenvector corresponding to the maximum eigenvalue
    max_index = np.argmax(eigenvalues)
    principal_eigenvector = eigenvectors[:, max_index].real # Use real part
    
    # Sort eigenvector components in descending order
    sorted_eigenvector = np.sort(principal_eigenvector)[::-1]
    
    # Project original data onto sorted eigenvector
    lambda_values = expert_data @ sorted_eigenvector
    
    # Normalize to get expert weights
    # Handle negative lambda values if they occur, e.g., by shifting
    if np.any(lambda_values < 0):
        lambda_values = lambda_values - np.min(lambda_values) 
        
    sum_lambda = np.sum(lambda_values)
    if sum_lambda == 0:
         # All experts are equal if sum is zero
        weights = np.ones(len(lambda_values)) / len(lambda_values)
    else:
        weights = lambda_values / sum_lambda
        
    return weights, eigenvalues[max_index].real, sorted_eigenvector, lambda_values

# ==================== DFS-AHP MODEL ====================

def dfs_ahp_aggregation(pairwise_matrices, expert_weights):
    """Aggregate DFS pairwise comparison matrices using DWGM"""
    n_criteria = len(pairwise_matrices[0])
    aggregated_matrix = [[None for _ in range(n_criteria)] for _ in range(n_criteria)]
    
    for i in range(n_criteria):
        for j in range(n_criteria):
            dfs_list = [matrix[i][j] for matrix in pairwise_matrices]
            aggregated_matrix[i][j] = dfs_dwgm(dfs_list, expert_weights)
    
    return aggregated_matrix

def compute_dfs_ahp_weights(aggregated_matrix):
    """Compute weights from aggregated DFS pairwise comparison matrix"""
    n = len(aggregated_matrix)
    
    # Compute geometric mean for each row using DWGM with equal weights
    geometric_means = []
    equal_weights = [1/n] * n
    
    for i in range(n):
        dfs_list = [aggregated_matrix[i][j] for j in range(n)]
        geometric_means.append(dfs_dwgm(dfs_list, equal_weights))
    
    # Sum of geometric means
    sum_geometric_means = geometric_means[0]
    for i in range(1, n):
        sum_geometric_means = dfs_addition(sum_geometric_means, geometric_means[i])
    
    # Calculate reciprocal of the sum
    # (1/O_a, 1/O_b) and (1/P_c, 1/P_d) are not standard DFS numbers.
    # We should perform division, which is complex.
    # A simpler approach (often used) is to defuzzify the GMs, sum them, and normalize.
    # But sticking to the paper's fuzzy ops: We need a fuzzy reciprocal.
    # Let's assume a simplified reciprocal for normalization
    
    sum_O_a, sum_O_b = sum_geometric_means['O']
    sum_P_c, sum_P_d = sum_geometric_means['P']

    # Simplified reciprocal (this is an approximation, real fuzzy division is complex)
    reciprocal_sum = {
        'O': (1/sum_O_a if sum_O_a != 0 else 0, 1/sum_O_b if sum_O_b != 0 else 0),
        'P': (1/sum_P_c if sum_P_c != 0 else 0, 1/sum_P_d if sum_P_d != 0 else 0)
    }

    # Normalize weights
    weights = []
    for i in range(n):
        # This multiplication might produce invalid DFS numbers if reciprocal isn't handled carefully.
        # Let's use the defuzzified-then-normalized approach for stability, as fuzzy-AHP normalization is tricky.
        pass # Will do defuzzified approach below

    # Alternative: Defuzzify geometric means and normalize (more stable)
    defuzzified_gms = [dfs_defuzzification(gm) for gm in geometric_means]
    sum_defuzzified_gms = sum(defuzzified_gms)
    
    if sum_defuzzified_gms == 0:
        normalized_weights_crisp = [1/n] * n
    else:
        normalized_weights_crisp = [gm / sum_defuzzified_gms for gm in defuzzified_gms]

    # Return both the fuzzy GMs (as 'weights') and the crisp normalized weights
    return geometric_means, normalized_weights_crisp


# ==================== DFS-QFD MODEL ====================

def dfs_qfd_relationship_matrix(relationships, expert_weights):
    """Aggregate DFS relationship matrices using DWGM"""
    n_rcs = len(relationships[0])
    n_mss = len(relationships[0][0])
    
    aggregated_matrix = [[None for _ in range(n_mss)] for _ in range(n_rcs)]
    
    for i in range(n_rcs):
        for j in range(n_mss):
            dfs_list = [expert_matrix[i][j] for expert_matrix in relationships]
            aggregated_matrix[i][j] = dfs_dwgm(dfs_list, expert_weights)
    
    return aggregated_matrix

def compute_dfs_qfd_scores(rc_weights_dfs, relationship_matrix):
    """Compute DFS scores for mitigation strategies"""
    n_rcs = len(rc_weights_dfs)
    n_mss = len(relationship_matrix[0])
    
    scores = [None] * n_mss
    
    for j in range(n_mss):
        dfs_list = []
        
        for i in range(n_rcs):
            # Multiply RC weight by relationship strength
            product = dfs_multiplication(rc_weights_dfs[i], relationship_matrix[i][j])
            dfs_list.append(product)
        
        # Sum the products using addition operation
        if dfs_list:
            total = dfs_list[0]
            for k in range(1, len(dfs_list)):
                total = dfs_addition(total, dfs_list[k])
            scores[j] = total
    
    return scores

# ==================== MILP MODEL ====================

def solve_milp_optimization(ms_scores, implementation_costs, implementation_times,
                            saving_costs, saving_times, available_budget, available_time):
    """
    Solve the MILP optimization problem using PuLP.
    """
    n_ms = len(ms_scores)
    
    # 1. Create the model
    model = pulp.LpProblem("Resilience_Strategy_Optimization", pulp.LpMaximize)
    
    # 2. Define Decision Variables
    # x[j] = 1 if mitigation strategy j is selected, 0 otherwise
    x = pulp.LpVariable.dicts("MS", range(n_ms), cat='Binary')
    
    # y[i][j] = 1 if both MS_i and MS_j are selected
    # This is to linearize the quadratic saving term (x_i * x_j)
    y = pulp.LpVariable.dicts("y",
                             ((i, j) for i in range(n_ms) for j in range(i + 1, n_ms)),
                             cat='Binary')
    
    # 3. Define Objective Function
    # Maximize total resilience score
    model += pulp.lpSum(ms_scores[j] * x[j] for j in range(n_ms)), "Total_Resilience_Score"
    
    # 4. Define Constraints
    
    # Cost Constraint
    total_implementation_cost = pulp.lpSum(implementation_costs[j] * x[j] for j in range(n_ms))
    total_cost_saving = pulp.lpSum(saving_costs[i][j] * y[(i, j)]
                                   for i in range(n_ms) for j in range(i + 1, n_ms))
    model += total_implementation_cost - total_cost_saving <= available_budget, "Budget_Constraint"
    
    # Time Constraint
    total_implementation_time = pulp.lpSum(implementation_times[j] * x[j] for j in range(n_ms))
    total_time_saving = pulp.lpSum(saving_times[i][j] * y[(i, j)]
                                   for i in range(n_ms) for j in range(i + 1, n_ms))
    model += total_implementation_time - total_time_saving <= available_time, "Time_Constraint"
    
    # Linearization Constraints for y[i][j] = x[i] * x[j]
    for i in range(n_ms):
        for j in range(i + 1, n_ms):
            model += y[(i, j)] <= x[i], f"Linearization_Constraint_y_{i}_{j}_1"
            model += y[(i, j)] <= x[j], f"Linearization_Constraint_y_{i}_{j}_2"
            model += y[(i, j)] >= x[i] + x[j] - 1, f"Linearization_Constraint_y_{i}_{j}_3"
            
    # 5. Solve the model
    # model.solve() uses the default CBC solver
    model.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 suppresses solver output
    
    # 6. Extract Results
    if pulp.LpStatus[model.status] == 'Optimal':
        selected_ms = [j for j in range(n_ms) if x[j].varValue > 0.5]
        
        # Calculate final values
        total_score = pulp.value(model.objective)
        total_cost = pulp.value(total_implementation_cost - total_cost_saving)
        total_time = pulp.value(total_implementation_time - total_time_saving)
        
        return selected_ms, total_score, total_cost, total_time
    else:
        # No optimal solution found (e.g., infeasible)
        return None, 0, 0, 0


# ==================== STREAMLIT APP ====================

def main():
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.markdown('<div class="logo">📊</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">Integrated DFS-AHP-QFD-MILP Decision Support System</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; color: #555;">
    <p style="font-size: 1.1rem;">This application implements an integrated decision support system with four modules:
    ML-based Expert Weighting, Decomposed Fuzzy AHP, Decomposed Fuzzy QFD, and MILP Optimization.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'expert_weights' not in st.session_state:
        st.session_state.expert_weights = None
    if 'rc_weights' not in st.session_state:
        st.session_state.rc_weights = None # This will store crisp weights
    if 'rc_weights_dfs' not in st.session_state:
        st.session_state.rc_weights_dfs = None # This will store fuzzy weights
    if 'ms_scores' not in st.session_state:
        st.session_state.ms_scores = None
    
    # Step navigation
    steps = [
        "ML-Based Expert Weighting",
        "Decomposed Fuzzy AHP", 
        "Decomposed Fuzzy QFD",
        "MILP Optimization",
        "Results Summary"
    ]
    
    # Use st.query_params to manage steps (more robust than session state alone for navigation)
    # This part is optional but good practice
    
    current_step_idx = st.session_state.current_step - 1
    
    # Progress bar
    progress = (st.session_state.current_step) / len(steps)
    st.progress(progress, text=f"Step {st.session_state.current_step}/{len(steps)}: {steps[current_step_idx]}")

    
    # Step 1: ML-Based Expert Weighting
    if st.session_state.current_step == 1:
        step1_ml_expert_weighting()
    
    # Step 2: Decomposed Fuzzy AHP
    elif st.session_state.current_step == 2:
        step2_dfs_ahp()
    
    # Step 3: Decomposed Fuzzy QFD
    elif st.session_state.current_step == 3:
        step3_dfs_qfd()
    
    # Step 4: MILP Optimization
    elif st.session_state.current_step == 4:
        step4_milp_optimization()
    
    # Step 5: Results Summary
    elif st.session_state.current_step == 5:
        step5_results_summary()

def step1_ml_expert_weighting():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Step 1: ML-Based Expert Weighting</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Instructions:</strong> Upload expert dimensional data or use the default example data. 
    The ML model will compute expert weights based on variance preservation.
    </div>
    """, unsafe_allow_html=True)
    
    # Option to use default data or upload custom data
    data_option = st.radio(
        "Select data source:",
        ["Use default example data", "Upload custom data"],
        horizontal=True
    )
    
    expert_data = None
    cov_matrix = None
    
    if data_option == "Use default example data":
        # Default expert data (4 experts × 10 dimensions)
        expert_data = np.array([
            [43.5, 19, 56500, 3, 4, 3, 9, 10, 3, 1000],  # Ex1
            [40.5, 16.5, 65000, 2, 3, 2, 8, 13, 2, 1000],  # Ex2
            [34.0, 11.0, 35500, 1, 3, 2, 3, 4, 3, 900],   # Ex3
            [38.0, 13.0, 45000, 2, 1, 2, 7, 5, 1, 500]    # Ex4
        ])
        
        # Default covariance matrix
        cov_matrix = np.array([
            [0.1343, 0.1390, 0.1153, 0.1250, 0.0570, 0.1184, 0.1338, 0.1155, 0.0066, 0.0526],
            [0.1390, 0.1492, 0.1237, 0.1250, 0.0820, 0.1289, 0.1322, 0.1302, 0.0332, 0.0828],
            [0.1153, 0.1237, 0.1441, 0.0890, 0.0480, 0.0508, 0.1222, 0.1516, -0.0148, 0.0720],
            [0.1250, 0.1250, 0.0890, 0.1250, 0.0417, 0.1250, 0.1250, 0.0833, 0.0000, 0.0250],
            [0.0570, 0.0820, 0.0480, 0.0417, 0.1319, 0.1042, 0.0243, 0.0741, 0.1354, 0.1417],
            [0.1184, 0.1289, 0.0508, 0.1250, 0.1042, 0.1875, 0.0938, 0.0556, 0.0938, 0.0750],
            [0.1338, 0.1322, 0.1222, 0.1250, 0.0243, 0.0938, 0.1441, 0.1157, -0.0365, 0.0208],
            [0.1155, 0.1302, 0.1516, 0.0833, 0.0741, 0.0556, 0.1157, 0.1667, 0.0139, 0.1056],
            [0.0066, 0.0332, -0.0148, 0.0000, 0.1354, 0.0938, -0.0365, 0.0139, 0.1719, 0.1375],
            [0.0526, 0.0828, 0.0720, 0.0250, 0.1417, 0.0750, 0.0208, 0.1056, 0.1375, 0.1700]
        ])
        
        st.success("Using default example data with 4 experts and 10 dimensional features.")
        
    else:
        # File upload for custom data
        uploaded_file = st.file_uploader("Upload expert data (CSV or Excel)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    expert_data_df = pd.read_csv(uploaded_file)
                else:
                    expert_data_df = pd.read_excel(uploaded_file)
                
                # Try to automatically skip index/header columns if they look like text
                if expert_data_df.iloc[:, 0].dtype == 'object':
                    expert_data_df = expert_data_df.set_index(expert_data_df.columns[0])
                    
                expert_data = expert_data_df.values
                cov_matrix = None  # Will be computed from data
                st.success(f"Loaded data with {expert_data.shape[0]} experts and {expert_data.shape[1]} features.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        else:
            st.info("Please upload a CSV or Excel file with expert dimensional data.")
            return
    
    # Display expert data
    st.subheader("Expert Dimensional Data")
    df_expert_data = pd.DataFrame(
        expert_data, 
        columns=[f"Feature {i+1}" for i in range(expert_data.shape[1])],
        index=[f"Expert {i+1}" for i in range(expert_data.shape[0])]
    )
    st.dataframe(df_expert_data, use_container_width=True)
    
    if st.button("Compute Expert Weights"):
        with st.spinner("Computing expert weights using ML model..."):
            try:
                expert_weights, max_eigenvalue, sorted_eigenvector, lambda_values = ml_expert_weighting(
                    expert_data, cov_matrix)
                
                # Store results in session state
                st.session_state.expert_weights = expert_weights
                st.session_state.ml_results = {
                    'max_eigenvalue': max_eigenvalue,
                    'sorted_eigenvector': sorted_eigenvector,
                    'lambda_values': lambda_values,
                    'expert_data': expert_data
                }
                
                # Display results
                st.subheader("ML Model Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Maximum Eigenvalue", f"{max_eigenvalue:.6f}")
                    st.write("Sorted Eigenvector Components:")
                    st.dataframe(pd.Series(sorted_eigenvector), use_container_width=True)
                
                with col2:
                    st.write("Expert Scores and Weights:")
                    results_df = pd.DataFrame({
                        'Expert': [f"Expert {i+1}" for i in range(len(expert_weights))],
                        'λ Score': lambda_values,
                        'Weight': expert_weights
                    })
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                st.success("Expert weights computed successfully!")
                
            except Exception as e:
                st.error(f"Error computing expert weights: {e}")
    
    if st.session_state.expert_weights is not None:
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Next: Decomposed Fuzzy AHP", key="step1_next"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.markdown("</div>", unsafe_allow_html=True)

def step2_dfs_ahp():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Step 2: Decomposed Fuzzy AHP</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Instructions:</strong> Define resilience challenges (criteria) and provide pairwise comparisons 
    using DFS linguistic terms. The system will compute weights for resilience challenges.
    </div>
    """, unsafe_allow_html=True)
    
    # Get expert weights from previous step
    if st.session_state.expert_weights is None:
        st.error("Please complete Step 1 first to compute expert weights.")
        if st.button("Go to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    n_experts = len(st.session_state.expert_weights)
    
    # Input for number of resilience challenges
    n_rc = st.number_input(
        "Number of Resilience Challenges (Criteria)",
        min_value=2,
        max_value=10,
        value=4,
        step=1
    )
    
    # Input resilience challenge names
    st.subheader("Resilience Challenge Names")
    rc_names = []
    cols = st.columns(n_rc)
    for i in range(n_rc):
        rc_name = cols[i].text_input(
            f"Resilience Challenge {i+1}",
            value=f"RC{i+1}",
            key=f"rc_name_{i}"
        )
        rc_names.append(rc_name)
    
    # Display DFS linguistic scale
    with st.expander("DFS Linguistic Scale Reference"):
        scale_df = pd.DataFrame([
            {"Linguistic Term": term, "O(μ,θ)": f"({values['O'][0]}, {values['O'][1]})", 
             "P(μ,θ)": f"({values['P'][0]}, {values['P'][1]})", "Meaning": term}
            for i, (term, values) in enumerate(dfs_linguistic_scale.items())
        ])
        st.dataframe(scale_df, use_container_width=True, hide_index=True)
    
    # Collect pairwise comparisons from each expert
    st.subheader("Pairwise Comparison Matrices")
    st.write("Each expert provides pairwise comparisons for resilience challenges. **You only need to fill the upper triangle.**")
    
    expert_matrices = []
    
    # Use tabs for experts
    expert_tabs = st.tabs([f"Expert {i+1} (Weight: {st.session_state.expert_weights[i]:.3f})" for i in range(n_experts)])
    
    for expert_idx in range(n_experts):
        with expert_tabs[expert_idx]:
            
            # Create an empty matrix for this expert
            expert_matrix = [[None for _ in range(n_rc)] for _ in range(n_rc)]
            
            # Create a dataframe for easier input
            comparison_data = []
            for i in range(n_rc):
                row = [None] * n_rc
                row[i] = "EEI"  # Diagonal
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(
                comparison_data,
                columns=rc_names,
                index=rc_names
            )
            
            # Use data editor for pairwise comparisons
            edited_df = st.data_editor(
                df_comparison,
                column_config={
                    col_name: st.column_config.SelectboxColumn(
                        col_name,
                        options=linguistic_options,
                        required=True,
                        default="EEI"
                    ) for col_name in rc_names
                },
                use_container_width=True,
                key=f"expert_{expert_idx}_comparisons"
            )
            
            # Convert the edited dataframe to DFS matrix
            all_filled = True
            for i in range(n_rc):
                for j in range(n_rc):
                    if i == j:
                        expert_matrix[i][j] = dfs_linguistic_scale['EEI']  # Exactly Equal on diagonal
                    elif i < j: # Only read upper triangle
                        term = edited_df.iloc[i, j]
                        if not term:
                            all_filled = False
                        expert_matrix[i][j] = dfs_linguistic_scale[term]
                    else: # Automatically fill lower triangle with reciprocal
                        reciprocal_term = edited_df.iloc[j, i]
                        original = dfs_linguistic_scale[reciprocal_term]
                        expert_matrix[i][j] = {
                            'O': original['P'], # Swap O and P
                            'P': original['O']
                        }
            
            expert_matrices.append(expert_matrix)
    
    if st.button("Compute Resilience Challenge Weights"):
        with st.spinner("Computing DFS-AHP weights..."):
            try:
                # Aggregate expert matrices
                aggregated_matrix = dfs_ahp_aggregation(expert_matrices, st.session_state.expert_weights)
                
                # Compute weights
                dfs_weights, normalized_weights = compute_dfs_ahp_weights(aggregated_matrix)
                
                # Store results
                st.session_state.rc_weights = normalized_weights # Crisp weights
                st.session_state.rc_weights_dfs = dfs_weights # Fuzzy weights (GMs)
                st.session_state.rc_names = rc_names
                st.session_state.dfs_ahp_results = {
                    'aggregated_matrix': aggregated_matrix,
                    'dfs_weights': dfs_weights,
                    'normalized_weights': normalized_weights
                }
                
                # Display results
                st.subheader("Resilience Challenge Weights")
                
                results_df = pd.DataFrame({
                    'Resilience Challenge': rc_names,
                    'DFS Weight (GM) - O': [f"({w['O'][0]:.3f}, {w['O'][1]:.3f})" for w in dfs_weights],
                    'DFS Weight (GM) - P': [f"({w['P'][0]:.3f}, {w['P'][1]:.3f})" for w in dfs_weights],
                    'CI': [dfs_consistency_index(w) for w in dfs_weights],
                    'Normalized Weight (Crisp)': normalized_weights
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Display ranking
                ranking_df = results_df.sort_values(by='Normalized Weight (Crisp)', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                ranking_df = ranking_df[['Rank', 'Resilience Challenge', 'Normalized Weight (Crisp)']]
                
                st.subheader("Ranking of Resilience Challenges")
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                
                st.success("Resilience challenge weights computed successfully!")
                
            except Exception as e:
                st.error(f"Error computing DFS-AHP weights: {e}")
                st.exception(e) # Show full traceback
    
    if st.session_state.rc_weights is not None:
        st.markdown("</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous: Expert Weighting", key="step2_prev"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("Next: Decomposed Fuzzy QFD", key="step2_next"):
                st.session_state.current_step = 3
                st.rerun()
    else:
        st.markdown("</div>", unsafe_allow_html=True)

def step3_dfs_qfd():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Step 3: Decomposed Fuzzy QFD</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Instructions:</strong> Define mitigation strategies and their relationships with resilience challenges.
    The system will compute priority scores for mitigation strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if previous steps are completed
    if st.session_state.rc_weights is None:
        st.error("Please complete Step 2 first to compute resilience challenge weights.")
        if st.button("Go to Step 2"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    rc_names = st.session_state.rc_names
    n_rc = len(rc_names)
    n_experts = len(st.session_state.expert_weights)
    
    # Input for number of mitigation strategies
    n_ms = st.number_input(
        "Number of Mitigation Strategies",
        min_value=2,
        max_value=15,
        value=5,
        step=1
    )
    
    # Input mitigation strategy names
    st.subheader("Mitigation Strategy Names")
    ms_names = []
    cols = st.columns(n_ms)
    for i in range(n_ms):
        ms_name = cols[i].text_input(
            f"Mitigation Strategy {i+1}",
            value=f"MS{i+1}",
            key=f"ms_name_{i}"
        )
        ms_names.append(ms_name)
    
    # Collect relationship assessments from experts
    st.subheader("Relationship Assessments")
    st.write("Experts assess the strength of relationship between resilience challenges (rows) and mitigation strategies (columns).")
    
    expert_relationships = []
    
    # Use tabs for experts
    expert_tabs = st.tabs([f"Expert {i+1} (Weight: {st.session_state.expert_weights[i]:.3f})" for i in range(n_experts)])
    
    for expert_idx in range(n_experts):
        with expert_tabs[expert_idx]:
            
            # Create relationship matrix for this expert
            relationship_matrix = [[None for _ in range(n_ms)] for _ in range(n_rc)]
            
            # Create a dataframe for input
            relationship_data = [["EEI" for _ in range(n_ms)] for _ in range(n_rc)]
            
            df_relationship = pd.DataFrame(
                relationship_data,
                columns=ms_names,
                index=rc_names
            )
            
            # Use data editor for relationship assessments
            edited_df = st.data_editor(
                df_relationship,
                column_config={
                    col: st.column_config.SelectboxColumn(
                        col,
                        options=linguistic_options,
                        required=True,
                        default="EEI"
                    ) for col in ms_names
                },
                use_container_width=True,
                key=f"expert_{expert_idx}_relationships"
            )
            
            # Convert to DFS values
            for i in range(n_rc):
                for j in range(n_ms):
                    term = edited_df.iloc[i, j]
                    if term:
                        relationship_matrix[i][j] = dfs_linguistic_scale[term]
                    else:
                        # Default to Equal Importance if no data
                        relationship_matrix[i][j] = dfs_linguistic_scale['EEI']
            
            expert_relationships.append(relationship_matrix)
    
    if st.button("Compute Mitigation Strategy Scores"):
        with st.spinner("Computing DFS-QFD scores..."):
            try:
                # Aggregate relationship matrices
                aggregated_relationships = dfs_qfd_relationship_matrix(
                    expert_relationships, st.session_state.expert_weights)
                
                # Get the fuzzy RC weights (geometric means from AHP)
                dfs_rc_weights = st.session_state.rc_weights_dfs
                
                # Compute MS scores
                ms_scores_dfs = compute_dfs_qfd_scores(dfs_rc_weights, aggregated_relationships)
                
                # Defuzzify scores
                ms_scores = [dfs_defuzzification(score) for score in ms_scores_dfs]
                
                # Compute AI_j (Aggregate Importance)
                AI_j = ms_scores
                
                # Store results
                st.session_state.ms_scores = AI_j # Store crisp scores for MILP
                st.session_state.ms_names = ms_names
                st.session_state.dfs_qfd_results = {
                    'aggregated_relationships': aggregated_relationships,
                    'dfs_scores': ms_scores_dfs,
                    'AI_j': AI_j
                }
                
                # Display results
                st.subheader("Mitigation Strategy Scores")
                
                results_df = pd.DataFrame({
                    'Mitigation Strategy': ms_names,
                    'DFS Score (O)': [f"({s['O'][0]:.3f}, {s['O'][1]:.3f})" for s in ms_scores_dfs],
                    'DFS Score (P)': [f"({s['P'][0]:.3f}, {s['P'][1]:.3f})" for s in ms_scores_dfs],
                    'CI': [dfs_consistency_index(s) for s in ms_scores_dfs],
                    'AI_j Score (Crisp)': AI_j
                })
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Display ranking
                ranking_df = results_df.sort_values(by='AI_j Score (Crisp)', ascending=False)
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                ranking_df = ranking_df[['Rank', 'Mitigation Strategy', 'AI_j Score (Crisp)']]
                
                st.subheader("Ranking of Mitigation Strategies")
                st.dataframe(ranking_df, use_container_width=True, hide_index=True)
                
                st.success("Mitigation strategy scores computed successfully!")
                
            except Exception as e:
                st.error(f"Error computing DFS-QFD scores: {e}")
                st.exception(e)
    
    if st.session_state.ms_scores is not None:
        st.markdown("</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous: DFS-AHP", key="step3_prev"):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("Next: MILP Optimization", key="step3_next"):
                st.session_state.current_step = 4
                st.rerun()
    else:
        st.markdown("</div>", unsafe_allow_html=True)

def step4_milp_optimization():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Step 4: MILP Optimization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Instructions:</strong> Provide implementation costs, times, and saving parameters for mitigation strategies.
    The system will optimize selection to maximize resilience performance within budget and time constraints.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if previous steps are completed
    if st.session_state.ms_scores is None:
        st.error("Please complete Step 3 first to compute mitigation strategy scores.")
        if st.button("Go to Step 3"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    ms_names = st.session_state.ms_names
    n_ms = len(ms_names)
    ms_scores = st.session_state.ms_scores
    
    # Input optimization parameters
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_budget = st.number_input(
            "Available Budget (ℸ)",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )
    
    with col2:
        available_time = st.number_input(
            "Available Time (∄, e.g., in months)",
            min_value=0.0,
            value=12.0,
            step=1.0
        )
    
    # Input implementation costs and times
    st.subheader("Implementation Costs and Times")
    
    implementation_costs = []
    implementation_times = []
    
    # Use data editor for faster input
    cost_time_df = pd.DataFrame(
        [
            {
                "Strategy": ms_name,
                "Cost": 10000.0 + i * 5000.0,
                "Time (months)": 3.0 + i * 1.0,
            }
            for i, ms_name in enumerate(ms_names)
        ]
    ).set_index("Strategy")

    edited_cost_time_df = st.data_editor(
        cost_time_df,
        column_config={
            "Cost": st.column_config.NumberColumn(format="$%.2f", min_value=0),
            "Time (months)": st.column_config.NumberColumn(min_value=0)
        },
        use_container_width=True
    )
    implementation_costs = edited_cost_time_df["Cost"].tolist()
    implementation_times = edited_cost_time_df["Time (months)"].tolist()

    
    # Input saving costs and times (interdependencies)
    st.subheader("Saving Parameters (Interdependencies)")
    st.write("Define cost and time savings when strategies are implemented together.")
    
    saving_costs = [[0.0 for _ in range(n_ms)] for _ in range(n_ms)]
    saving_times = [[0.0 for _ in range(n_ms)] for _ in range(n_ms)]
    
    # Create a list of pairs for easier input
    pairs = [(i, j) for i in range(n_ms) for j in range(i+1, n_ms)]
    if not pairs:
        st.info("At least two mitigation strategies are needed to define interdependencies.")
    else:
        # Data editor for savings
        savings_data = [
            {
                "Pair": f"{ms_names[i]} - {ms_names[j]}",
                "Cost Saving": 1000.0,
                "Time Saving": 0.5,
                "_pair_idx": (i, j) # Helper to map back
            }
            for i, j in pairs
        ]
        
        edited_savings_df = st.data_editor(
            savings_data,
            column_config={
                "Pair": st.column_config.TextColumn(disabled=True),
                "Cost Saving": st.column_config.NumberColumn(min_value=0.0, step=100.0),
                "Time Saving": st.column_config.NumberColumn(min_value=0.0, step=0.1),
                "_pair_idx": None # Hide this column
            },
            hide_index=True,
            use_container_width=True
        )

        for item in edited_savings_df:
            i, j = item["_pair_idx"]
            saving_costs[i][j] = item["Cost Saving"]
            saving_costs[j][i] = item["Cost Saving"] # Symmetric
            saving_times[i][j] = item["Time Saving"]
            saving_times[j][i] = item["Time Saving"] # Symmetric
            
    
    if st.button("Solve MILP Optimization"):
        with st.spinner("Solving MILP optimization problem..."):
            try:
                # Solve the MILP problem
                selected_ms, total_score, total_cost, total_time = solve_milp_optimization(
                    ms_scores, implementation_costs, implementation_times,
                    saving_costs, saving_times, available_budget, available_time
                )
                
                if selected_ms is None:
                    st.error("Optimization failed. The problem is likely infeasible.")
                    st.warning("This means no combination of strategies can satisfy your budget and time constraints. Try increasing the available budget or time.")
                    return

                # Store results
                st.session_state.milp_results = {
                    'selected_ms': selected_ms,
                    'total_score': total_score,
                    'total_cost': total_cost,
                    'total_time': total_time,
                    'available_budget': available_budget,
                    'available_time': available_time,
                    'implementation_costs': implementation_costs,
                    'implementation_times': implementation_times
                }
                
                # Display results
                st.subheader("Optimization Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Resilience Score", f"{total_score:.3f}")
                
                with col2:
                    st.metric("Total Cost", f"${total_cost:,.2f}")
                
                with col3:
                    st.metric("Total Time", f"{total_time:.1f} months")
                
                with col4:
                    st.metric("Selected Strategies", len(selected_ms))
                
                # Display selected strategies
                st.subheader("Selected Mitigation Strategies")
                if selected_ms:
                    selected_df = pd.DataFrame({
                        'Mitigation Strategy': [ms_names[i] for i in selected_ms],
                        'Resilience Score': [ms_scores[i] for i in selected_ms],
                        'Cost': [implementation_costs[i] for i in selected_ms],
                        'Time': [implementation_times[i] for i in selected_ms]
                    })
                    st.dataframe(selected_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No strategies were selected. The constraints might be too tight, or no strategy provides positive value.")
                
                # Display resource utilization
                st.subheader("Resource Utilization")
                col1, col2 = st.columns(2)
                
                with col1:
                    budget_utilization = (total_cost / available_budget) * 100 if available_budget > 0 else 0
                    st.metric("Budget Utilization", f"{budget_utilization:.1f}%")
                    st.progress(budget_utilization / 100)
                
                with col2:
                    time_utilization = (total_time / available_time) * 100 if available_time > 0 else 0
                    st.metric("Time Utilization", f"{time_utilization:.1f}%")
                    st.progress(time_utilization / 100)
                
                st.success("MILP optimization completed successfully!")
                
            except Exception as e:
                st.error(f"Error solving MILP optimization: {e}")
                st.exception(e)
    
    if hasattr(st.session_state, 'milp_results'):
        st.markdown("</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous: DFS-QFD", key="step4_prev"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("Next: Results Summary", key="step4_next"):
                st.session_state.current_step = 5
                st.rerun()
    else:
        st.markdown("</div>", unsafe_allow_html=True)

def step5_results_summary():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Step 5: Results Summary</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="instruction-box">
    <strong>Summary:</strong> This page provides a comprehensive overview of all analysis results.
    </div>
    """, unsafe_allow_html=True)
    
    # Check if all steps are completed
    if not all(k in st.session_state for k in ['expert_weights', 'rc_weights', 'ms_scores', 'milp_results']):
        st.error("Not all steps have been completed. Please go back and complete the analysis.")
        if st.button("Go to Step 1"):
            st.session_state.current_step = 1
            st.rerun()
        return

    col1, col2 = st.columns(2)

    with col1:
        # Expert Weights Summary
        if hasattr(st.session_state, 'expert_weights'):
            st.subheader("Expert Weights (Step 1)")
            expert_df = pd.DataFrame({
                'Expert': [f"Expert {i+1}" for i in range(len(st.session_state.expert_weights))],
                'Weight': st.session_state.expert_weights
            })
            st.dataframe(expert_df, use_container_width=True, hide_index=True)
        
        # Resilience Challenge Weights Summary
        if hasattr(st.session_state, 'rc_weights'):
            st.subheader("Resilience Challenge Weights (Step 2)")
            rc_df = pd.DataFrame({
                'Resilience Challenge': st.session_state.rc_names,
                'Weight': st.session_state.rc_weights
            }).sort_values('Weight', ascending=False).reset_index(drop=True)
            rc_df['Rank'] = range(1, len(rc_df) + 1)
            st.dataframe(rc_df[['Rank', 'Resilience Challenge', 'Weight']], 
                         use_container_width=True, hide_index=True)
        
        # Mitigation Strategy Scores Summary
        if hasattr(st.session_state, 'ms_scores'):
            st.subheader("Mitigation Strategy Scores (Step 3)")
            ms_df = pd.DataFrame({
                'Mitigation Strategy': st.session_state.ms_names,
                'AI_j Score': st.session_state.ms_scores
            }).sort_values('AI_j Score', ascending=False).reset_index(drop=True)
            ms_df['Rank'] = range(1, len(ms_df) + 1)
            st.dataframe(ms_df[['Rank', 'Mitigation Strategy', 'AI_j Score']], 
                         use_container_width=True, hide_index=True)
    
    with col2:
        # MILP Optimization Results Summary
        if hasattr(st.session_state, 'milp_results'):
            st.subheader("Optimization Results (Step 4)")
            milp_results = st.session_state.milp_results
            
            st.metric("Total Resilience Score", f"{milp_results['total_score']:.3f}")
            
            c1, c2 = st.columns(2)
            c1.metric("Total Cost", f"${milp_results['total_cost']:,.2f}",
                      f"out of ${milp_results['available_budget']:,.2f}")
            c2.metric("Total Time", f"{milp_results['total_time']:.1f} months",
                      f"out of {milp_results['available_time']:.1f} months")

            st.metric("Selected Strategies", len(milp_results['selected_ms']))
            
            if milp_results['selected_ms']:
                st.write("**Optimal Set of Mitigation Strategies:**")
                selected_list = [st.session_state.ms_names[i] for i in milp_results['selected_ms']]
                
                selected_final_df = pd.DataFrame({
                    'Strategy': selected_list,
                    'Score': [st.session_state.ms_scores[i] for i in milp_results['selected_ms']],
                    'Cost': [milp_results['implementation_costs'][i] for i in milp_results['selected_ms']],
                    'Time': [milp_results['implementation_times'][i] for i in milp_results['selected_ms']]
                }).sort_values('Score', ascending=False)

                st.dataframe(selected_final_df, use_container_width=True, hide_index=True)

    # Recommendations
    st.subheader("Recommendations")
    
    if (hasattr(st.session_state, 'milp_results') and 
        st.session_state.milp_results['selected_ms']):
        
        st.markdown("""
        <div class="success-box">
        <strong>Implementation Priority:</strong>
        <ul>
            <li>Focus on implementing the <b>optimal set</b> of mitigation strategies listed above.</li>
            <li>The selection is not just based on individual scores but on the <b>best combination</b> that respects interdependencies and constraints.</li>
            <li>Monitor budget and time utilization throughout implementation.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        <strong>Recommendations:</strong>
        <ul>
            <li><b>No optimal solution found or no strategies selected.</b></li>
            <li>Review the constraints (Step 4) and consider adjusting budget or time allocations.</li>
            <li>Re-evaluate the mitigation strategy costs and implementation times.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous: MILP Optimization", key="step5_prev"):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.button("Start New Analysis", key="step5_new"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.current_step = 1
            st.rerun()

if __name__ == "__main__":
    main()
