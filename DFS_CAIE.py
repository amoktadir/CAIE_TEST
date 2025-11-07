import streamlit as st
import pandas as pd
import numpy as np
from math import prod

class DecomposedFuzzyAHP:
    def __init__(self):
        # DFS linguistic scale mapping
        self.dfs_scale = {
            'EMI': {'mu_O': 0.90, 'nu_O': 0.10, 'mu_P': 0.10, 'nu_P': 0.90},
            'PMI': {'mu_O': 0.85, 'nu_O': 0.15, 'mu_P': 0.15, 'nu_P': 0.85},
            'AMI': {'mu_O': 0.80, 'nu_O': 0.20, 'mu_P': 0.20, 'nu_P': 0.80},
            'VSI': {'mu_O': 0.75, 'nu_O': 0.25, 'mu_P': 0.25, 'nu_P': 0.75},
            'StMI': {'mu_O': 0.70, 'nu_O': 0.30, 'mu_P': 0.30, 'nu_P': 0.70},
            'MI': {'mu_O': 0.65, 'nu_O': 0.35, 'mu_P': 0.35, 'nu_P': 0.65},
            'WMI': {'mu_O': 0.60, 'nu_O': 0.40, 'mu_P': 0.40, 'nu_P': 0.60},
            'SMI': {'mu_O': 0.55, 'nu_O': 0.45, 'mu_P': 0.45, 'nu_P': 0.55},
            'EEI': {'mu_O': 0.50, 'nu_O': 0.50, 'mu_P': 0.50, 'nu_P': 0.50},
            'SMU': {'mu_O': 0.45, 'nu_O': 0.55, 'mu_P': 0.55, 'nu_P': 0.45},
            'WMU': {'mu_O': 0.40, 'nu_O': 0.60, 'mu_P': 0.60, 'nu_P': 0.40},
            'MU': {'mu_O': 0.35, 'nu_O': 0.65, 'mu_P': 0.65, 'nu_P': 0.35},
            'StMU': {'mu_O': 0.30, 'nu_O': 0.70, 'mu_P': 0.70, 'nu_P': 0.30},
            'VSU': {'mu_O': 0.25, 'nu_O': 0.75, 'mu_P': 0.75, 'nu_P': 0.25},
            'AMU': {'mu_O': 0.20, 'nu_O': 0.80, 'mu_P': 0.80, 'nu_P': 0.20},
            'PMU': {'mu_O': 0.15, 'nu_O': 0.85, 'mu_P': 0.85, 'nu_P': 0.15},
            'EMU': {'mu_O': 0.10, 'nu_O': 0.90, 'mu_P': 0.90, 'nu_P': 0.10}
        }
        
        # Linguistic term descriptions
        self.term_descriptions = {
            'EMI': 'Exactly More Important',
            'PMI': 'Perfectly More Important',
            'AMI': 'Absolutely More Important',
            'VSI': 'Very Strongly More Important',
            'StMI': 'Strongly More Important',
            'MI': 'More Important',
            'WMI': 'Weakly More Important',
            'SMI': 'Slightly More Important',
            'EEI': 'Exactly Equal Importance',
            'SMU': 'Slightly More Unimportant',
            'WMU': 'Weakly More Unimportant',
            'MU': 'More Unimportant',
            'StMU': 'Strongly More Unimportant',
            'VSU': 'Very Strongly More Unimportant',
            'AMU': 'Absolutely More Unimportant',
            'PMU': 'Perfectly More Unimportant',
            'EMU': 'Exactly More Unimportant'
        }

    def get_dfs_value(self, term):
        """Get DFS values for a linguistic term"""
        return self.dfs_scale.get(term, self.dfs_scale['EEI'])

    def dwgm_operator(self, dfs_values, weights):
        """Decomposed Weighted Geometric Mean operator"""
        n = len(dfs_values)
        mu_O = prod(dfs_values[i]['mu_O'] ** weights[i] for i in range(n))
        nu_O = prod(dfs_values[i]['nu_O'] ** weights[i] for i in range(n))
        mu_P = prod(dfs_values[i]['mu_P'] ** weights[i] for i in range(n))
        nu_P = prod(dfs_values[i]['nu_P'] ** weights[i] for i in range(n))
        
        return {'mu_O': mu_O, 'nu_O': nu_O, 'mu_P': mu_P, 'nu_P': nu_P}

    def defuzzify(self, dfs_number):
        """Defuzzify DFS number using Eq. (12)"""
        mu_O = dfs_number['mu_O']
        nu_O = dfs_number['nu_O']
        mu_P = dfs_number['mu_P']
        nu_P = dfs_number['nu_P']
        
        return (mu_O + mu_P - nu_O - nu_P + 1) / 3

def main():
    st.set_page_config(page_title="Decomposed Fuzzy AHP", layout="wide")
    
    st.title("🧮 Decomposed Fuzzy Analytic Hierarchy Process (DFS-AHP)")
    st.markdown("""
    This application implements the Decomposed Fuzzy Sets based AHP method for multi-criteria decision making.
    Based on the methodology from Cebi et al. (2022, 2023) and Tüysüz & Kahraman (2024).
    """)
    
    # Initialize DFS-AHP processor
    dfs_ahp = DecomposedFuzzyAHP()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio(
        "Select Section:",
        ["Introduction", "Linguistic Scale", "Pairwise Comparisons", "Aggregation & Results"]
    )
    
    if app_section == "Introduction":
        st.header("📚 Introduction to DFS-AHP")
        st.markdown("""
        ### Decomposed Fuzzy Sets (DFS)
        
        DFS is the latest extension of intuitionistic fuzzy sets that captures uncertainty and vagueness 
        by allowing experts to express preferences in both optimistic and pessimistic terms.
        """)
        
    elif app_section == "Linguistic Scale":
        st.header("🗣️ DFS Linguistic Scale")
        
        # Display the linguistic scale table
        scale_data = []
        for term, values in dfs_ahp.dfs_scale.items():
            if term in dfs_ahp.term_descriptions:
                scale_data.append({
                    'Linguistic Term': f"{term} ({dfs_ahp.term_descriptions[term]})",
                    'μ_O': values['mu_O'],
                    'ν_O': values['nu_O'],
                    'μ_P': values['mu_P'],
                    'ν_P': values['nu_P']
                })
        
        scale_df = pd.DataFrame(scale_data)
        st.dataframe(scale_df, use_container_width=True)
        
    elif app_section == "Pairwise Comparisons":
        st.header("⚖️ Pairwise Comparisons")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Setup")
            num_criteria = st.number_input("Number of Criteria/RCs", min_value=2, max_value=10, value=3)
            num_experts = st.number_input("Number of Experts", min_value=1, max_value=5, value=1)
            
            # Criteria names
            st.subheader("Criteria Names")
            criteria_names = []
            for i in range(num_criteria):
                name = st.text_input(f"Criterion {i+1} Name", value=f"RC{i+1}")
                criteria_names.append(name)
        
        with col2:
            st.subheader("Pairwise Comparison Matrix")
            st.info("Select linguistic terms for each pairwise comparison")
            
            # Initialize session state for storing comparisons
            if 'comparisons' not in st.session_state:
                st.session_state.comparisons = {}
            
            # Create comparison matrix for each expert
            for expert in range(num_experts):
                st.markdown(f"### Expert {expert + 1}")
                
                # Create comparison matrix
                comparison_matrix = []
                for i in range(num_criteria):
                    row = []
                    cols = st.columns(num_criteria + 1)
                    cols[0].write(f"**{criteria_names[i]}**")
                    
                    for j in range(num_criteria):
                        if i == j:
                            row.append('EEI')
                            cols[j+1].write("EEI")
                        else:
                            # Get available terms (optimistic for upper triangle, pessimistic for lower)
                            available_terms = list(dfs_ahp.dfs_scale.keys())
                            if i < j:
                                # Upper triangle - optimistic terms
                                available_terms = [t for t in available_terms if t in ['EMI', 'PMI', 'AMI', 'VSI', 'StMI', 'MI', 'WMI', 'SMI', 'EEI']]
                            else:
                                # Lower triangle - pessimistic terms  
                                available_terms = [t for t in available_terms if t in ['EMU', 'PMU', 'AMU', 'VSU', 'StMU', 'MU', 'WMU', 'SMU', 'EEU']]
                            
                            selected_term = cols[j+1].selectbox(
                                f"{criteria_names[i]} vs {criteria_names[j]}",
                                available_terms,
                                key=f"expert_{expert}_{i}_{j}"
                            )
                            row.append(selected_term)
                    comparison_matrix.append(row)
                
                st.session_state.comparisons[f'expert_{expert}'] = {
                    'matrix': comparison_matrix,
                    'criteria_names': criteria_names
                }
                
                st.markdown("---")
    
    elif app_section == "Aggregation & Results":
        st.header("📊 Aggregation & Results")
        
        if 'comparisons' not in st.session_state or not st.session_state.comparisons:
            st.warning("Please complete pairwise comparisons first.")
            return
        
        # Get expert weights
        st.subheader("Expert Weights")
        num_experts = len(st.session_state.comparisons)
        expert_weights = []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Assign Expert Weights")
            weight_sum = 0
            for i in range(num_experts):
                weight = st.slider(f"Expert {i+1} Weight", 0.1, 1.0, 1.0/num_experts, key=f"weight_{i}")
                expert_weights.append(weight)
                weight_sum += weight
            
            # Normalize weights
            if weight_sum > 0:
                expert_weights = [w/weight_sum for w in expert_weights]
            st.write(f"Normalized weights: {[f'{w:.3f}' for w in expert_weights]}")
        
        with col2:
            # Calculate aggregated weights
            st.markdown("### Calculate Aggregated Weights")
            if st.button("Calculate Final Weights"):
                all_expert_weights = []
                criteria_names = st.session_state.comparisons['expert_0']['criteria_names']
                num_criteria = len(criteria_names)
                
                # Process each expert's comparisons
                for expert_idx in range(num_experts):
                    expert_data = st.session_state.comparisons[f'expert_{expert_idx}']
                    comparison_matrix = expert_data['matrix']
                    
                    # Convert linguistic terms to DFS values for each criterion
                    criterion_dfs_values = []
                    for i in range(num_criteria):
                        criterion_comparisons = []
                        for j in range(num_criteria):
                            term = comparison_matrix[i][j]
                            dfs_value = dfs_ahp.get_dfs_value(term)
                            criterion_comparisons.append(dfs_value)
                        criterion_dfs_values.append(criterion_comparisons)
                    
                    # Calculate weights for this expert using DWGM
                    expert_criterion_weights = []
                    for i in range(num_criteria):
                        # Use equal weights for aggregation within criterion
                        equal_weights = [1.0/num_criteria] * num_criteria
                        aggregated_dfs = dfs_ahp.dwgm_operator(criterion_dfs_values[i], equal_weights)
                        defuzzified_weight = dfs_ahp.defuzzify(aggregated_dfs)
                        expert_criterion_weights.append(defuzzified_weight)
                    
                    # Normalize weights for this expert
                    total = sum(expert_criterion_weights)
                    if total > 0:
                        expert_criterion_weights = [w/total for w in expert_criterion_weights]
                    
                    all_expert_weights.append(expert_criterion_weights)
                
                # Aggregate across experts using DWGM
                final_weights = []
                for criterion_idx in range(num_criteria):
                    criterion_expert_values = []
                    for expert_idx in range(num_experts):
                        dfs_representation = {
                            'mu_O': all_expert_weights[expert_idx][criterion_idx],
                            'nu_O': 1 - all_expert_weights[expert_idx][criterion_idx],
                            'mu_P': all_expert_weights[expert_idx][criterion_idx],
                            'nu_P': 1 - all_expert_weights[expert_idx][criterion_idx]
                        }
                        criterion_expert_values.append(dfs_representation)
                    
                    # Use expert weights for aggregation
                    aggregated_dfs = dfs_ahp.dwgm_operator(criterion_expert_values, expert_weights)
                    final_weight = dfs_ahp.defuzzify(aggregated_dfs)
                    final_weights.append(final_weight)
                
                # Normalize final weights
                total_final = sum(final_weights)
                if total_final > 0:
                    final_weights = [w/total_final for w in final_weights]
                
                # Display results
                st.subheader("Final Weights")
                results_data = []
                for i, (criterion, weight) in enumerate(zip(criteria_names, final_weights)):
                    results_data.append({
                        'Criterion': criterion,
                        'Weight': weight,
                        'Rank': i + 1
                    })
                
                results_df = pd.DataFrame(results_data)
                results_df = results_df.sort_values('Weight', ascending=False)
                results_df['Weight'] = results_df['Weight'].round(4)
                results_df['Rank'] = range(1, len(results_df) + 1)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization using Streamlit native charts
                st.subheader("Visualization")
                
                # Bar chart using Streamlit native
                st.bar_chart(results_df.set_index('Criterion')['Weight'])
                
                # Display as metric cards
                st.subheader("Weight Distribution")
                cols = st.columns(3)
                for i, row in results_df.iterrows():
                    with cols[i % 3]:
                        st.metric(
                            label=row['Criterion'],
                            value=f"{row['Weight']:.3f}",
                            delta=f"Rank: {row['Rank']}"
                        )

if __name__ == "__main__":
    main()
