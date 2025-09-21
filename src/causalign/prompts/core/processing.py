# src/causalign/prompts/core/processing.py

import itertools

import pandas as pd


# abc version
def expand_domain_to_dataframe(domain_dict):
    """
    Expand a domain dictionary into a DataFrame with 8 unique counterbalance condition combinations.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary created using create_domain_dict() function.

    Returns:
    --------
    pd.DataFrame
        DataFrame where each row represents a unique combination of variable values.
    """
    domain_name = domain_dict["domain_name"]
    variables = domain_dict["variables"]

    # Prepare data storage
    data = []
    var_keys = []  # Stores variable keys (A, B, C)
    var_names = []  # Stores actual variable names (e.g., "interest rates")
    var_p_values = []
    var_m_values = []

    # Extract variable details and prepare p/m values separately
    for var, details in variables.items():
        var_name = details["name"]  # Fetch actual variable name
        var_detailed = details["detailed"]

        p_values = list(details["p_value"].items())
        m_values = (
            list(details["m_value"].items()) if "m_value" in details else p_values
        )

        var_keys.append(var)
        var_names.append(var_name)  # Store the actual variable name
        var_p_values.append(p_values)
        var_m_values.append(m_values)

    # Construct all 2^3 = 8 unique counterbalance conditions
    cntbl_conditions = list(itertools.product(["p", "m"], repeat=3))

    for cntbl_combo in cntbl_conditions:
        row = {"domain": domain_name}
        cntbl_cond = ""  # Counterbalance condition initialization

        for var, var_name, cntbl, p_vals, m_vals in zip(
            var_keys, var_names, cntbl_combo, var_p_values, var_m_values
        ):
            # Select value based on counterbalancing condition
            chosen_values = p_vals if cntbl == "p" else m_vals
            var_value, var_sense = chosen_values[0]  # Take first valid pair

            row[f"{var}"] = var_name  # Assign actual variable name
            row[f"{var}_values"] = var_value
            row[f"{var}_cntbl"] = cntbl
            row[f"{var}_sense"] = var_sense
            row[f"{var}_detailed"] = var_detailed

            cntbl_cond += cntbl  # Concatenate all counterbalance conditions

        row["cntbl_cond"] = cntbl_cond
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure all 8 unique sequences exist
    assert sorted(df["cntbl_cond"].unique()) == sorted(
        ["ppp", "ppm", "pmp", "pmm", "mpp", "mpm", "mmp", "mmm"]
    ), f"Expected all 8 unique sequences, but got: {df['cntbl_cond'].unique()}"

    return df


# C1, C2, E version
# def expand_df_by_task_queries(df, inference_tasks):
#     """
#     Expand the dataframe with verbalized inference tasks.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame created using expand_domain_to_dataframe.
#     inference_tasks : dict
#         Dictionary of inference tasks mapping.

#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame where each row represents a unique combination of variable values and an inference task.
#     """
#     expanded_rows = []

#     for _, row in df.iterrows():
#         for task, task_info in inference_tasks.items():
#             # For tasks a-h and j, do both (C1, C2) and (C2, C1)
#             if task in {"a", "b", "c", "d", "e", "f", "g", "h", "j"}:
#                 for Ci, Cj in [("C1", "C2"), ("C2", "C1")]:
#                     new_row = row.copy()
#                     new_row["task"] = task
#                     new_row["query_node"] = task_info["query_node"].replace("Ci", Ci)
#                     new_row["observation"] = (
#                         task_info["observation"].replace("Ci", Ci).replace("Cj", Cj)
#                     )
#                     new_row["query"] = (
#                         task_info["query"].replace("Ci", Ci).replace("Cj", Cj)
#                     )
#                     expanded_rows.append(new_row)

#             # For tasks i and k, only do one assignment
#             else:
#                 new_row = row.copy()
#                 new_row["task"] = task
#                 new_row["query_node"] = (
#                     task_info["query_node"].replace("Ci", "C1").replace("Cj", "C2")
#                 )
#                 new_row["observation"] = (
#                     task_info["observation"].replace("Ci", "C1").replace("Cj", "C2")
#                 )
#                 new_row["query"] = (
#                     task_info["query"].replace("Ci", "C1").replace("Cj", "C2")
#                 )
#                 expanded_rows.append(new_row)

#     # Create expanded DataFrame
#     expanded_df = pd.DataFrame(expanded_rows)


#     return expanded_df


# ########## a, b, c version ########
# def expand_df_by_task_queries(df, inference_tasks):
#     """
#     Expand the dataframe with verbalized inference tasks.

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame created using expand_domain_to_dataframe.
#     inference_tasks : dict
#         Dictionary of inference tasks mapping.

#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame where each row represents a unique combination of variable values and an inference task.
#     """
#     expanded_rows = []

#     for _, row in df.iterrows():
#         for task, task_info in inference_tasks.items():
#             # For tasks a-h and j, do both (A, B) and (B, A)
#             if task in {"a", "b", "c", "d", "e", "f", "g", "h", "j"}:
#                 for Ai, Bj in [("A", "B"), ("B", "A")]:
#                     new_row = row.copy()
#                     new_row["task"] = task
#                     new_row["query_node"] = task_info["query_node"].replace("Ai", Ai)
#                     new_row["observation"] = (
#                         task_info["observation"].replace("Ai", Ai).replace("Bj", Bj)
#                     )
#                     new_row["query"] = (
#                         task_info["query"].replace("Ai", Ai).replace("Bj", Bj)
#                     )
#                     expanded_rows.append(new_row)

#             # For tasks i and k, only do one assignment
#             else:
#                 new_row = row.copy()
#                 new_row["task"] = task
#                 new_row["query_node"] = (
#                     task_info["query_node"].replace("Ai", "A").replace("Bj", "B")
#                 )
#                 new_row["observation"] = (
#                     task_info["observation"].replace("Ai", "A").replace("Bj", "B")
#                 )
#                 new_row["query"] = (
#                     task_info["query"].replace("Ai", "A").replace("Bj", "B")
#                 )
#                 expanded_rows.append(new_row)

#     # Create expanded DataFrame
#     expanded_df = pd.DataFrame(expanded_rows)

#     return expanded_df


##########x,y,z version ########
def expand_df_by_task_queries(df, inference_tasks):
    """
    Expand the dataframe with verbalized inference tasks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame created using expand_domain_to_dataframe.
    inference_tasks : dict
        Dictionary of inference tasks mapping.

    Returns:
    --------
    pd.DataFrame
        DataFrame where each row represents a unique combination of variable values and an inference task.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        for task, task_info in inference_tasks.items():
            # For tasks a-h and j, do both (A, B) and (B, A)
            if task in {"a", "b", "c", "d", "e", "f", "g", "h", "j"}:
                for Xi, Yj in [("X", "Y"), ("Y", "X")]:
                    new_row = row.copy()
                    new_row["task"] = task
                    new_row["query_node"] = task_info["query_node"].replace("Xi", Xi)
                    new_row["observation"] = (
                        task_info["observation"].replace("Xi", Xi).replace("Yj", Yj)
                    )
                    new_row["query"] = (
                        task_info["query"].replace("Xi", Xi).replace("Yj", Yj)
                    )
                    expanded_rows.append(new_row)

            # For tasks i and k, only do one assignment
            else:
                new_row = row.copy()
                new_row["task"] = task
                new_row["query_node"] = (
                    task_info["query_node"].replace("Xi", "X").replace("Yj", "Y")
                )
                new_row["observation"] = (
                    task_info["observation"].replace("Xi", "X").replace("Yj", "Y")
                )
                new_row["query"] = (
                    task_info["query"].replace("Xi", "X").replace("Yj", "Y")
                )
                expanded_rows.append(new_row)

    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df


def generate_prompt_dataframe(
    domain_dict,
    inference_tasks,
    graph_type,
    graph_structures,
    counterbalance_enabled=True,
    prompt_category="numeric",
    prompt_type="Please provide only a numeric response and no additional information",
    indep_causes_collider=False,
):
    """
    Generate a complete prompt dataframe with verbalized prompts.

    This function combines domain expansion, task expansion, and prompt verbalization
    to create a complete dataset ready for LLM querying.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary containing domain information created by create_domain_dict()
    inference_tasks : dict
        A dictionary containing inference tasks mapping
    graph_type : str
        The type of graph structure (e.g., "collider", "fork", "chain")
    graph_structures : dict
        A dictionary containing graph structure templates
    counterbalance_enabled : bool
        Whether to include counterbalance conditions (default: True)
    prompt_category : str
        Category identifier for the prompt style
        prompt_type : str
        Instructions for how the LLM should respond
    indep_causes_collider : bool
        Whether causes in collider graphs are independent (default: False)

    Returns:
    --------
    pd.DataFrame
        A DataFrame with expanded prompts and graph structure information
    """
    from .verbalization import (
        verbalize_causal_mechanism,
        verbalize_domain_intro,
        verbalize_inference_task,
        verbalize_variables_section,
    )

    # Step 1: Expand the domain dictionary into a DataFrame
    # containing all unique combinations of variable values
    df = expand_df_by_task_queries(
        expand_domain_to_dataframe(domain_dict), inference_tasks
    )

    # Step 2: Extract the domain introduction text
    domain_intro = verbalize_domain_intro(domain_dict)

    # Step 3: Add graph type to the DataFrame
    df["graph"] = graph_type

    # Step 4: Generate the full prompt by combining all sections
    df["prompt"] = df.apply(
        lambda row: (
            domain_intro
            + " Here are some variables: "
            + verbalize_variables_section(domain_dict, row)
            + " Here are the causal relationships: "
            + verbalize_causal_mechanism(
                domain_dict, row, graph_type, graph_structures, indep_causes_collider
            )
            + verbalize_inference_task(
                row, nested_dict=domain_dict, prompt_type=prompt_type
            )
        ),
        axis=1,
    )

    # Step 5: Add prompt category
    df["prompt_category"] = prompt_category

    return df
