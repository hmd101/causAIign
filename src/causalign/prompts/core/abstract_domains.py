"""
Abstract Domain Definitions

Defines abstract and fantasy domains for LLM reasoning experiments
without human baseline data.
"""

# Abstract Mathematical Domain
abstract_math_domain = {
    "domain_name": "abstract_math",
    "introduction": "Mathematicians study abstract relationships between symbolic entities. In this domain, we examine how mathematical properties influence each other through logical connections.",
    "variables": {
        "X": {
            "name": "alpha property",
            "detailed": "The alpha property is a fundamental characteristic that entities can possess.",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Y": {
            "name": "beta property",
            "detailed": "The beta property is another fundamental characteristic that entities can possess.",
            "p_value": {"1": "present", "0": "absent"},
            "m_value": {"1": "absent", "0": "present"},
        },
        "Z": {
            "name": "gamma outcome",
            "detailed": "The gamma outcome is the result that emerges from the interaction of properties.",
            "p_value": {"1": "positive", "0": "negative"},
            "m_value": {"1": "negative", "0": "positive"},
        },
    },
}

# Fantasy World Domain
fantasy_domain = {
    "domain_name": "fantasy",
    "introduction": "In the mystical realm of Aetheria, magical scholars study the relationships between different forms of energy and their effects on the natural world.",
    "variables": {
        "X": {
            "name": "crystal energy",
            "detailed": "Crystal energy is the mystical force that flows through enchanted crystals found throughout the realm.",
            "p_value": {"1": "radiant", "0": "low"},
            "m_value": {"1": "dim", "0": "radiant"},
        },
        "Y": {
            "name": "moon phase alignment",
            "detailed": "Moon phase alignment refers to how well the celestial bodies are positioned to channel magical forces.",
            "p_value": {"1": "harmonious", "0": "weak"},
            "m_value": {"1": "discordant", "0": "harmonious"},
        },
        "Z": {
            "name": "spell potency",
            "detailed": "Spell potency is the strength and effectiveness of magical incantations cast in the realm.",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}

abstract_abc_domain = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables A, B, and C.",
    "variables": {
        "X": {
            "name": "A",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "B",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "C",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}

abstract_xyz_domain = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables X, Y, and Z.",
    "variables": {
        "X": {
            "name": "X",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "Y",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "Z",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}

abstract_vhl_domain = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables V, H, and L.",
    "variables": {
        "X": {
            "name": "V",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "H",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "L",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}

abstract_triplets_domain = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic triplets XWL, QLO, and KZW.",
    "variables": {
        "X": {
            "name": "XWL",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "QLO",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "KZW",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}


abstract_triplets_domain_symb = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic triplets &%#, {*$ and <!}",
    "variables": {
        "X": {
            "name": "&%#",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "{*$",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "<!}",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}


abstract_quintuplets_domain = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic quintuplets 1HIF*, F&DFK, and KKX@J.",
    "variables": {
        "X": {
            "name": "1HIF*",
            "detailed": "",
            "p_value": {"1": "high", "0": "low"},
            "m_value": {"1": "low", "0": "high"},
        },
        "Y": {
            "name": "F&DFK",
            "detailed": "",
            "p_value": {"1": "strong", "0": "weak"},
            "m_value": {"1": "weak", "0": "strong"},
        },
        "Z": {
            "name": "KK1@J",
            "detailed": "",
            "p_value": {"1": "powerful", "0": "weak"},
            "m_value": {"1": "weak", "0": "powerful"},
        },
    },
}



################################
# abstract domains with randomly generated strings
################################
abs_alnum_10 ={
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables u8jzPde0Ig, xLd6GncfBA, and epfJBd0Kh8.",
    "variables": {
        "X": {
            "name": "u8jzPde0Ig",
            "detailed": "",
            "p_value": {
                "1": "high",
                "0": "low"
            },
            "m_value": {
                "1": "low",
                "0": "high"
            }
        },
        "Y": {
            "name": "xLd6GncfBA",
            "detailed": "",
            "p_value": {
                "1": "strong",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "strong"
            }
        },
        "Z": {
            "name": "epfJBd0Kh8",
            "detailed": "",
            "p_value": {
                "1": "powerful",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "powerful"
            }
        }
    }
}

abs_all_10 = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables PtY@gj'mU-, h#Bel31iEl, and )2h+pC==-h.",
    "variables": {
        "X": {
            "name": "PtY@gj'mU-",
            "detailed": "",
            "p_value": {
                "1": "high",
                "0": "low"
            },
            "m_value": {
                "1": "low",
                "0": "high"
            }
        },
        "Y": {
            "name": "h#Bel31iEl",
            "detailed": "",
            "p_value": {
                "1": "strong",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "strong"
            }
        },
        "Z": {
            "name": ")2h+pC==-h",
            "detailed": "",
            "p_value": {
                "1": "powerful",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "powerful"
            }
        }
    }
}


abs_num_symb_10 = {
    "domain_name": "systems",
    "introduction": "In abstract reasoning studies, researchers examine relationships between symbolic variables +9:~34]6.`, 3[$25<;4&5, and ^<3_7%}}`3.",
    "variables": {
        "X": {
            "name": "+9:~34]6.`",
            "detailed": "",
            "p_value": {
                "1": "high",
                "0": "low"
            },
            "m_value": {
                "1": "low",
                "0": "high"
            }
        },
        "Y": {
            "name": "3[$25<;4&5",
            "detailed": "",
            "p_value": {
                "1": "strong",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "strong"
            }
        },
        "Z": {
            "name": "^<3_7%}}`3",
            "detailed": "",
            "p_value": {
                "1": "powerful",
                "0": "weak"
            },
            "m_value": {
                "1": "weak",
                "0": "powerful"
            }
        }
    }
}
################################
################################
# Registry of abstract domains
ABSTRACT_DOMAINS = {
    "abs_alnum_10": abs_alnum_10,
    "abs_all_10": abs_all_10,
    "abs_num_symb_10": abs_num_symb_10,
    # "abstract_math": abstract_math_domain,
    # "abstract_abc": abstract_abc_domain,
    # "abstract_xyz": abstract_xyz_domain,
    # "abstract_vhl": abstract_vhl_domain,
    # "abstract_triplets": abstract_triplets_domain,
    # "abstract_triplets_symb": abstract_triplets_domain_symb,
    # "abstract_quintuplets": abstract_quintuplets_domain,
}

FANTASY_DOMAINS = {
 # "fantasy": fantasy_domain,
}