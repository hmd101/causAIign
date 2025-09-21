# src/causalign/prompts/core/verbalization.py


### xyz version


def capitalize_after_periods(text):
    """
    Capitalize the first letter after each period in a text string, and also
    capitalize the very first letter of the text.

    Parameters:
    -----------
    text : str
        The input text to process

    Returns:
    --------
    str
        Text with proper capitalization after periods and at the beginning
    """
    if not isinstance(text, str) or not text:
        return text

    # Split by periods, capitalize first letter of each sentence
    sentences = text.split(". ")

    # Process all sentences (including the first one)
    for i in range(len(sentences)):
        sentence = sentences[i].strip()
        if sentence:
            # Capitalize first character if it's a letter
            sentences[i] = (
                sentence[0].upper() + sentence[1:]
                if len(sentence) > 1
                else sentence.upper()
            )

    # Rejoin with '. '
    return ". ".join(sentences)


def verbalize_domain_intro(domain_dict):
    """
    Extract and return the domain introduction text.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary containing domain information.

    Returns:
    --------
    str
        The introduction text of the domain.
    """
    # return domain_dict["domain"]["introduction"]
    intro = domain_dict["introduction"]
    return capitalize_after_periods(intro)


def verbalize_causal_mechanism(
    domain_dict, row, graph_type, graph_structures, indep_causes_collider
):
    """
    Create a detailed description of causal relationships with explanations.

    Parameters:
    -----------
    domain_dict : dict
        The domain dictionary
    row : pd.Series
        Row from the expanded dataframe with counterbalance information
    graph_type : str
        Type of causal graph (collider, fork, chain)
    graph_structures : dict
        Graph structure templates
    indep_causes_collider : bool
        Whether the causes are independent variables, ie whether each cause
        can independently cause the outcome


    Returns:
    --------
    str
        Formatted text describing causal relationships with explanations
    """
    if graph_type not in graph_structures:
        return ""

    domain_name = domain_dict["domain_name"].upper()
    causal_text = (
        " "  # f"\n{domain_name}\n********** CAUSAL RELATIONSHIPS **********\n"
    )

    # Get variables and their counterbalance conditions
    x_cntbl = row["X_cntbl"]
    y_cntbl = row["Y_cntbl"]
    z_cntbl = row["Z_cntbl"]

    # Get variable senses
    x_sense = row["X_sense"]
    y_sense = row["Y_sense"]
    z_sense = row["Z_sense"]

    # Get variable names
    x_name = row["X"]
    y_name = row["Y"]
    z_name = row["Z"]

    # Build causal relationships based on graph type
    if graph_type == "collider":
        # X → Z relationship with explanation
        x_z_relation = f"{x_sense} {x_name} causes {z_sense} {z_name}. "

        # Get X → Z explanation
        x_z_key = f"{x_cntbl}_{z_cntbl}"
        x_z_explanation = ""
        if (
            "explanations" in domain_dict["variables"]["X"]
            and x_z_key in domain_dict["variables"]["X"]["explanations"]
        ):
            explanation_text = domain_dict["variables"]["X"]["explanations"][x_z_key]
            x_z_explanation = " " + capitalize_after_periods(explanation_text)

        # Y → Z relationship with explanation
        y_z_relation = f"{y_sense} {y_name} causes {z_sense} {z_name}."

        # Get Y → Z explanation
        y_z_key = f"{y_cntbl}_{z_cntbl}"
        y_z_explanation = ""
        if (
            "explanations" in domain_dict["variables"]["Y"]
            and y_z_key in domain_dict["variables"]["Y"]["explanations"]
        ):
            explanation_text = domain_dict["variables"]["Y"]["explanations"][y_z_key]
            y_z_explanation = " " + capitalize_after_periods(explanation_text)

        # Combine relationships and explanations
        # If the causes are independent variables, ie whether each cause
        # can independently cause the outcome, then we need to add the
        # explanation for the other cause
        if indep_causes_collider:
            indep_cause_explanation = (
                f"Both {x_sense} {x_name} and {y_sense} {y_name} can each cause {z_sense} {z_name} on their own. "
                + f"That is, it's not the case that both of these have to be present in order for {z_sense} {z_name} to be present. "
                + f"Rather, {x_sense} {x_name} can independently cause {z_sense} {z_name} on their own, and {y_sense} {y_name} can independently cause {z_sense} {z_name} on their own as well."
            )
            causal_text += f"{x_z_relation}{x_z_explanation} "
            causal_text += f"{y_z_relation}{y_z_explanation} {indep_cause_explanation} "
        else:
            causal_text += f"{x_z_relation}{x_z_explanation} "
            causal_text += f"{y_z_relation}{y_z_explanation} "

    # Add similar handling for fork and chain graphs if needed

    return capitalize_after_periods(causal_text)


def verbalize_inference_task(row, nested_dict, prompt_type):
    """
    Generate the verbalized inference task based on a DataFrame row.

    Parameters:
    -----------
    row : pd.Series
        A row from the DataFrame containing the inference task information.
    nested_dict : dict
        The nested dictionary containing value mappings for variables created by `create_domain_dict`.
    prompt_type : str
        The type of prompt to be displayed to the LLM.

    Returns:
    --------
    str
        The verbalized inference task statement.
    """
    variables_dict = nested_dict["variables"]
    observations = row["observation"].split(", ")

    # Set domain-specific observation intro
    domain = row["domain"].lower()
    if domain == "weather":
        observation_text = (
            "Suppose that there is a weather system that is known to have "
        )
    elif domain == "economy":
        observation_text = "Suppose that the economy is currently known to have "
    elif domain == "sociology":
        observation_text = (
            "Suppose that the society you live in currently exhibits the following: "
        )
    else:
        observation_text = "You are currently observing: "

    observation_descriptions = []

    for obs in observations:
        var_name, value = obs.split("=")
        cntbl_type = row[f"{var_name}_cntbl"]

        if var_name not in variables_dict:
            raise KeyError(f"Variable {var_name} not found in nested_dict['variables']")

        value_mapping = variables_dict[var_name]

        if f"{cntbl_type}_value" in value_mapping:
            sense_label = value_mapping[f"{cntbl_type}_value"].get(value)
        else:
            available_mapping = value_mapping.get(
                "p_value", value_mapping.get("m_value")
            )
            sense_label = available_mapping.get(value)

        if sense_label is None:
            raise KeyError(
                f"Value {value} for {var_name} is missing in nested_dict['variables'][{var_name}]"
            )

        observation_descriptions.append(f"{sense_label} {row[var_name]}")

    observation_text += " and ".join(observation_descriptions) + "."

    # Extract the query variable and construct the query
    query_var = row["query_node"].split("=")[0]
    query_sense = row[f"{query_var}_sense"]
    query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} is present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely likely. {prompt_type}"

    return capitalize_after_periods(observation_text + query_text)


def verbalize_variables_section(domain_dict, row):
    """
    Create a comprehensive description of all variables based on counterbalance conditions.

    Parameters:
    -----------
    domain_dict : dict
        The domain dictionary containing variable information
    row : pd.Series
        Row from the expanded dataframe with counterbalance information

    Returns:
    --------
    str
        Formatted text describing all variables with their counterbalanced values
    """
    domain_name = domain_dict["domain_name"]

    # replace domain name such that it fits into the sentence
    if domain_name == "economy":
        domain_name = "economies"

    elif domain_name == "sociology":
        domain_name = "societies"

    elif domain_name == "weather":
        domain_name = "weather systems"

    else:
        domain_name = "systems"

    variables_text = " "  # f"\n{domain_name}\n********** VARIABLES **********\n"

    for var_key, var_details in domain_dict["variables"].items():
        name = var_details["name"]
        detailed = var_details["detailed"]

        # Get appropriate value based on counterbalance
        cntbl = row[f"{var_key}_cntbl"]
        value_dict = var_details["p_value"] if cntbl == "p" else var_details["m_value"]

        # Create description with on/off values
        value_1 = value_dict["1"]
        value_0 = value_dict["0"]

        # Format the description
        variables_text += f"{detailed} Some {domain_name} have {value_1} {name}. Others have {value_0} {name}. "  # this should be different -> needed to reproduce exact RW17 prompts, but not needed for abstract prompts
        # else, for abstract prompts, it should be:
        # variables_text += f"{detailed} Sometimes {name} is {value_1} and sometimes{name} is {value_0}. "

    return capitalize_after_periods(variables_text)
