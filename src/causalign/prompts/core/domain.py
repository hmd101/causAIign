# src/causalign/prompts/core/domain.py



def create_domain_dict(
    domain_name, introduction, variables_config, graph_type="collider"
):
    """
    Create a domain dictionary with full support for explanations and counterbalance conditions.

    Parameters:
    -----------
    domain_name : str
        Domain name (e.g., "economy", "sociology")

    introduction : str
        Domain introduction text

    variables_config : dict
        Complete configuration for all variables (X, Y, Z):
        {
            "X": {
                "name": "interest rates",
                "detailed": "Interest rates are the rates banks charge...",
                "p_value": {"1": "low", "0": "normal"},
                "m_value": {"1": "high", "0": "normal"},
                "explanations": {
                    "p_p": "Low interest rates stimulate economic growth...",
                    "p_m": "The good economic times produced by...",
                    "m_p": "The high interest rates result in high yields...",
                    "m_m": "A lot of people are making large monthly interest..."
                }
            },
            "Y": {...},
            "Z": {...}
        }

    graph_type : str
        Type of causal graph (collider, fork, chain)

    Returns:
    --------
    dict
        Domain dictionary in the required format
    """
    # Start with basic structure
    domain_dict = {
        "domain_name": domain_name,
        "introduction": introduction,
        "variables": {},
        "graph_type": graph_type,
    }

    # Process each variable
    for var_key, config in variables_config.items():
        # Validate required fields
        required_fields = ["name", "detailed", "p_value"]
        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field '{field}' for variable {var_key}"
                )

        # Create variable entry
        var_entry = {
            "name": config["name"],
            "detailed": config["detailed"],
            "p_value": config["p_value"].copy(),
        }

        # Add m_value if provided, otherwise use opposite of p_value
        if "m_value" in config:
            var_entry["m_value"] = config["m_value"].copy()
        else:
            # Default behavior: swap 0/1 values from p_value
            var_entry["m_value"] = {
                "1": config["p_value"]["0"],
                "0": config["p_value"]["1"],
            }

        # Add explanations if provided
        if "explanations" in config:
            var_entry["explanations"] = config["explanations"].copy()

        # Add to domain_dict
        domain_dict["variables"][var_key] = var_entry

    return domain_dict
