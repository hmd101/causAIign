# src/causalign/prompts/core/constants.py

# # Contains default domain components used in rehder & Waldmann, 2017 (RW17) and graph structures


# rw_17_domain_components = {
#     "economy": {
#         "domain_name": "economy",
#         "variables": {
#             "C1": {
#                 "C1_name": "interest rates",
#                 "C1_detailed": "Interest rates are the rates banks charge to loan money.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Low interest rates stimulate economic growth, leading to greater prosperity overall, and allowing more money to be saved for retirement in particular.",
#                     "p_m": "The good economic times produced by the low interest rates leads to greater confidence and less worry about the future, so people are less concerned about retirement.",
#                     "m_p": "The high interest rates result in high yields on government bonds, which are especially attractive for retirement savings because they are such a safe investment.",
#                     "m_m": "A lot of people are making large monthly interest payments on credit card debt, and they have no money left to save for retirement.",
#                 },
#             },
#             "C2": {
#                 "C2_name": "trade deficits",
#                 "C2_detailed": "A country's trade deficit is the difference between the value of the goods that a country imports and the value of the goods that a country exports.",
#                 "p_value": {"1": "small", "0": "normal"},
#                 "m_value": {"1": "large", "0": "normal"},
#                 "explanations": {
#                     "p_p": "When the economy is good, people can cover their basic expenses and so have enough money left over to contribute to their retirement accounts.",
#                     "p_m": "When the economy is good, people are optimistic and so spend rather than save.",
#                     "m_p": "People become nervous when their economy is no longer competitive enough in the world economy to export products, and begin saving for retirement as a result.",
#                     "m_m": "The loss of local manufacturing jobs means that there are people out of work, and contributions to retirement accounts decreases.",
#                 },
#             },
#             "E": {
#                 "E_name": "retirement savings",
#                 "E_detailed": "Retirement savings is the money people save for their retirement.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Economists seek to describe and predict the regular patterns of economic fluctuation. To do this, they study some important variables or attributes of economies. They also study how these attributes are responsible for producing or causing one another.",
#     },
#     "sociology": {
#         "domain_name": "sociology",
#         "variables": {
#             "C1": {
#                 "C1_name": "urbanization",
#                 "C1_detailed": "Urbanization is the degree to which the members of a society live in urban environments (i.e., cities) versus rural environments.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Big cities provide many opportunities for financial and social improvement.",
#                     "p_m": "In big cities many people are competing for the same high-status jobs and occupations.",
#                     "m_p": 'People in rural areas are rarely career oriented, and so take time off from working and switch frequently between different "temp" jobs.',
#                     "m_m": "The low density of people prevents the dynamic economic expansion needed for people to get ahead.",
#                 },
#             },
#             "C2": {
#                 "C2_name": "interest in religion",
#                 "C2_detailed": "Interest in religion is the degree to which the members of a society show a curiosity in religion issues or participate in organized religions.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Without the restraint of religion-based morality, the impulse toward greed dominates and people tend to accumulate material wealth.",
#                     "p_m": "Many religions reinforce a strong work ethic; without this motivation, workers become complacent at their jobs.",
#                     "m_p": "Religion fosters communal care, and those of the same religion tend to support each other with jobs, financial favors, and so on.",
#                     "m_m": "The spiritualism induced by religion works to reduce the desire for material wealth.",
#                 },
#             },
#             "E": {
#                 "E_name": "socio-economic mobility",
#                 "E_detailed": "Socioeconomic mobility is the degree to which the members of a society are able to improve their social and economic status.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Sociologists seek to describe and predict the regular patterns of societal interactions. To do this, they study some important variables or attributes of societies. They also study how these attributes are responsible for producing or causing one another.",
#     },
#     "weather": {
#         "domain_name": "weather",
#         "variables": {
#             "C1": {
#                 "C1_name": "ozone levels",
#                 "C1_detailed": "Ozone is a gaseous allotrope of oxygen (O3) and is formed by exposure to UV radiation.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 # Note: this is graph dependent. the explanations below are based on the collider graph
#                 "explanations": {
#                     "p_p": "Ozone attracts extra oxygen atoms from water molecules, creating a concentration of water vapor in that region.",
#                     "p_m": "Ozone accepts extra oxygen atoms, decreasing the amount of oxygen available to form water molecules. With fewer water molecules, there is lower humidity.",
#                     "m_p": "The oxygen atoms that would normally be part of ozone molecules are free to combine with hydrogen atoms instead, creating water molecules.",
#                     "m_m": "The low amount of ozone allows a large number of ultra-violet (UV) rays to enter the atmosphere, and the UV rays break up water molecules, resulting in low humidity.",
#                 },
#             },
#             "C2": {
#                 "C2_name": "air pressure",
#                 "C2_detailed": "Air pressure is force exerted due to concentrations of air molecules.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#                 # Note: this is graph dependent. the explanations below are based on the collider graph
#                 "explanations": {
#                     "p_p": "When pressure does not force water vapor to break into oxygen and hydrogen atoms, water vapor remains in abundance.",
#                     "p_m": "Low air pressure poorly facilitates condensation; as a result, there are less water molecules in the air.",
#                     "m_p": "The higher pressure means that the components of water molecules (hydrogen and oxygen) tend to not dissocciate from one another. Because there are more water molecules, humidity is higher.",
#                     "m_m": "When air pressure is high, water vapor condenses into liquid water (rain), and the atmosphere is left with little moisture.",
#                 },
#             },
#             "E": {
#                 "E_name": "humidity",
#                 "E_detailed": "Humidity is the degree to which the atmosphere contains water molecules.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Meteorologists seek to describe and predict the regular patterns that govern weather systems. To do this, they study some important variables or attributes of weather systems. They also study how these attributes are responsible for producing or causing one another.",
#     },
# }


# #  graph structures (currently only collider in dataset, but expandable), RW17 used collider and fork.
# graph_structures = {
#     "collider": {
#         "description": "C1→E←C2",  # RW17: A→C←B
#         "causal_template": "{c1_sense} {c1_name} causes {e_sense} {e_name}. Also, {c2_sense} {c2_name} causes {e_sense} {e_name}.",
#     },
#     "fork": {
#         "description": "C2←C1→E",  # RW17: C←A→B
#         "causal_template": "{c1_sense} {c1_name} causes {c2_sense} {c2_name}. Also, {c1_sense} {c1_name} causes {e_sense} {e_name}.",
#     },
#     "chain": {
#         "description": "C1→C2→E",  # RW17: A→B→C
#         "causal_template": "{c1_sense} {c1_name} causes {c2_sense} {c2_name}. And {c2_sense} {c2_name} causes {e_sense} {e_name}.",
#     },
# }


# # Add inference task mapping to the structure
# inference_tasks_rw17 = {
#     "a": {
#         "query_node": "Ci=1",
#         "observation": "E=1, Cj=1",
#         "query": "p(Ci=1|E=1, Cj=1)",
#     },
#     "b": {"query_node": "Ci=1", "observation": "E=1", "query": "p(Ci=1|E=1)"},
#     "c": {
#         "query_node": "Ci=1",
#         "observation": "E=1, Cj=0",
#         "query": "p(Ci=1|E=1, Cj=0)",
#     },
#     "d": {"query_node": "Ci=1", "observation": "Cj=1", "query": "p(Ci=1|Cj=1)"},
#     "e": {"query_node": "Ci=1", "observation": "Cj=0", "query": "p(Ci=1|Cj=0)"},
#     "f": {
#         "query_node": "Ci=1",
#         "observation": "E=0, Cj=1",
#         "query": "p(Ci=1|E=0, Cj=1)",
#     },
#     "g": {"query_node": "Ci=1", "observation": "E=0", "query": "p(Ci=1|E=0)"},
#     "h": {
#         "query_node": "Ci=1",
#         "observation": "E=0, Cj=0",
#         "query": "p(Ci=1|E=0, Cj=0)",
#     },
#     "i": {
#         "query_node": "E=1",
#         "observation": "Ci=0, Cj=0",
#         "query": "p(E=1|Ci=0, Cj=0)",
#     },
#     "j": {
#         "query_node": "E=1",
#         "observation": "Ci=0, Cj=1",
#         "query": "p(E=1|Ci=0, Cj=1)",
#     },
#     "k": {
#         "query_node": "E=1",
#         "observation": "Ci=1, Cj=1",
#         "query": "p(E=1|Ci=1, Cj=1)",
#     },
# }


# # Contains default domain components used in rehder & Waldmann, 2017 (RW17) and graph structures
# ############# ABC version ##############

# rw_17_domain_components = {
#     "economy": {
#         "domain_name": "economy",
#         "variables": {
#             "A": {
#                 "A_name": "interest rates",
#                 "A_detailed": "Interest rates are the rates banks charge to loan money.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Low interest rates stimulate economic growth, leading to greater prosperity overall, and allowing more money to be saved for retirement in particular.",
#                     "p_m": "The good economic times produced by the low interest rates leads to greater confidence and less worry about the future, so people are less concerned about retirement.",
#                     "m_p": "The high interest rates result in high yields on government bonds, which are especially attractive for retirement savings because they are such a safe investment.",
#                     "m_m": "A lot of people are making large monthly interest payments on credit card debt, and they have no money left to save for retirement.",
#                 },
#             },
#             "B": {
#                 "B_name": "trade deficits",
#                 "B_detailed": "A country's trade deficit is the difference between the value of the goods that a country imports and the value of the goods that a country exports.",
#                 "p_value": {"1": "small", "0": "normal"},
#                 "m_value": {"1": "large", "0": "normal"},
#                 "explanations": {
#                     "p_p": "When the economy is good, people can cover their basic expenses and so have enough money left over to contribute to their retirement accounts.",
#                     "p_m": "When the economy is good, people are optimistic and so spend rather than save.",
#                     "m_p": "People become nervous when their economy is no longer competitive enough in the world economy to export products, and begin saving for retirement as a result.",
#                     "m_m": "The loss of local manufacturing jobs means that there are people out of work, and contributions to retirement accounts decreases.",
#                 },
#             },
#             "C": {
#                 "C_name": "retirement savings",
#                 "C_detailed": "Retirement savings is the money people save for their retirement.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Economists seek to describe and predict the regular patterns of economic fluctuation. To do this, they study some important variables or attributes of economies. They also study how these attributes are responsible for producing or causing one another.",
#     },
#     "sociology": {
#         "domain_name": "sociology",
#         "variables": {
#             "A": {
#                 "A_name": "urbanization",
#                 "A_detailed": "Urbanization is the degree to which the members of a society live in urban environments (i.e., cities) versus rural environments.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Big cities provide many opportunities for financial and social improvement.",
#                     "p_m": "In big cities many people are competing for the same high-status jobs and occupations.",
#                     "m_p": 'People in rural areas are rarely career oriented, and so take time off from working and switch frequently between different "temp" jobs.',
#                     "m_m": "The low density of people prevents the dynamic economic expansion needed for people to get ahead.",
#                 },
#             },
#             "B": {
#                 "B_name": "interest in religion",
#                 "B_detailed": "Interest in religion is the degree to which the members of a society show a curiosity in religion issues or participate in organized religions.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 "explanations": {
#                     "p_p": "Without the restraint of religion-based morality, the impulse toward greed dominates and people tend to accumulate material wealth.",
#                     "p_m": "Many religions reinforce a strong work ethic; without this motivation, workers become complacent at their jobs.",
#                     "m_p": "Religion fosters communal care, and those of the same religion tend to support each other with jobs, financial favors, and so on.",
#                     "m_m": "The spiritualism induced by religion works to reduce the desire for material wealth.",
#                 },
#             },
#             "C": {
#                 "C_name": "socio-economic mobility",
#                 "C_detailed": "Socioeconomic mobility is the degree to which the members of a society are able to improve their social and economic status.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Sociologists seek to describe and predict the regular patterns of societal interactions. To do this, they study some important variables or attributes of societies. They also study how these attributes are responsible for producing or causing one another.",
#     },
#     "weather": {
#         "domain_name": "weather",
#         "variables": {
#             "A": {
#                 "A_name": "ozone levels",
#                 "A_detailed": "Ozone is a gaseous allotrope of oxygen (O3) and is formed by exposure to UV radiation.",
#                 "p_value": {"1": "low", "0": "normal"},
#                 "m_value": {"1": "high", "0": "normal"},
#                 # Note: this is graph dependent. the explanations below are based on the collider graph
#                 "explanations": {
#                     "p_p": "Ozone attracts extra oxygen atoms from water molecules, creating a concentration of water vapor in that region.",
#                     "p_m": "Ozone accepts extra oxygen atoms, decreasing the amount of oxygen available to form water molecules. With fewer water molecules, there is lower humidity.",
#                     "m_p": "The oxygen atoms that would normally be part of ozone molecules are free to combine with hydrogen atoms instead, creating water molecules.",
#                     "m_m": "The low amount of ozone allows a large number of ultra-violet (UV) rays to enter the atmosphere, and the UV rays break up water molecules, resulting in low humidity.",
#                 },
#             },
#             "B": {
#                 "B_name": "air pressure",
#                 "B_detailed": "Air pressure is force exerted due to concentrations of air molecules.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#                 # Note: this is graph dependent. the explanations below are based on the collider graph
#                 "explanations": {
#                     "p_p": "When pressure does not force water vapor to break into oxygen and hydrogen atoms, water vapor remains in abundance.",
#                     "p_m": "Low air pressure poorly facilitates condensation; as a result, there are less water molecules in the air.",
#                     "m_p": "The higher pressure means that the components of water molecules (hydrogen and oxygen) tend to not dissocciate from one another. Because there are more water molecules, humidity is higher.",
#                     "m_m": "When air pressure is high, water vapor condenses into liquid water (rain), and the atmosphere is left with little moisture.",
#                 },
#             },
#             "C": {
#                 "C_name": "humidity",
#                 "C_detailed": "Humidity is the degree to which the atmosphere contains water molecules.",
#                 "p_value": {"1": "high", "0": "normal"},
#                 "m_value": {"1": "low", "0": "normal"},
#             },
#         },
#         "introduction": "Meteorologists seek to describe and predict the regular patterns that govern weather systems. To do this, they study some important variables or attributes of weather systems. They also study how these attributes are responsible for producing or causing one another.",
#     },
# }


# #  graph structures (currently only collider in dataset, but expandable), RW17 used collider and fork.
# graph_structures = {
#     "collider": {
#         "description": "A→C←B",  # RW17: A→C←B
#         "causal_template": "{a_sense} {a_name} causes {c_sense} {c_name}. Also, {b_sense} {b_name} causes {c_sense} {c_name}.",
#     },
#     "fork": {
#         "description": "B←A→C",  # RW17: C←A→B
#         "causal_template": "{a_sense} {a_name} causes {b_sense} {b_name}. Also, {a_sense} {a_name} causes {c_sense} {c_name}.",
#     },
#     "chain": {
#         "description": "A→B→C",  # RW17: A→B→C
#         "causal_template": "{a_sense} {a_name} causes {b_sense} {b_name}. And {b_sense} {b_name} causes {c_sense} {c_name}.",
#     },
# }


# # Add inference task mapping to the structure
# inference_tasks_rw17 = {
#     "a": {
#         "query_node": "Ai=1",
#         "observation": "C=1, Bj=1",
#         "query": "p(Ai=1|C=1, Bj=1)",
#     },
#     "b": {"query_node": "Ai=1", "observation": "C=1", "query": "p(Ai=1|C=1)"},
#     "c": {
#         "query_node": "Ai=1",
#         "observation": "C=1, Bj=0",
#         "query": "p(Ai=1|C=1, Bj=0)",
#     },
#     "d": {"query_node": "Ai=1", "observation": "Bj=1", "query": "p(Ai=1|Bj=1)"},
#     "e": {"query_node": "Ai=1", "observation": "Bj=0", "query": "p(Ai=1|Bj=0)"},
#     "f": {
#         "query_node": "Ai=1",
#         "observation": "C=0, Bj=1",
#         "query": "p(Ai=1|C=0, Bj=1)",
#     },
#     "g": {"query_node": "Ai=1", "observation": "C=0", "query": "p(Ai=1|C=0)"},
#     "h": {
#         "query_node": "Ai=1",
#         "observation": "C=0, Bj=0",
#         "query": "p(Ai=1|C=0, Bj=0)",
#     },
#     "i": {
#         "query_node": "C=1",
#         "observation": "Ai=0, Bj=0",
#         "query": "p(C=1|Ai=0, Bj=0)",
#     },
#     "j": {
#         "query_node": "C=1",
#         "observation": "Ai=0, Bj=1",
#         "query": "p(C=1|Ai=0, Bj=1)",
#     },
#     "k": {
#         "query_node": "C=1",
#         "observation": "Ai=1, Bj=1",
#         "query": "p(C=1|Ai=1, Bj=1)",
#     },
# }


# Contains default domain components used in rehder & Waldmann, 2017 (RW17) and graph structures
############# XYZ version ##############

rw_17_domain_components = {
    "economy": {
        "domain_name": "economy",
        "variables": {
            "X": {
                "name": "interest rates",
                "detailed": "Interest rates are the rates banks charge to loan money.",
                "p_value": {"1": "low", "0": "normal"},
                "m_value": {"1": "high", "0": "normal"},
                "explanations": {
                    "p_p": "Low interest rates stimulate economic growth, leading to greater prosperity overall, and allowing more money to be saved for retirement in particular.",
                    "p_m": "The good economic times produced by the low interest rates leads to greater confidence and less worry about the future, so people are less concerned about retirement.",
                    "m_p": "The high interest rates result in high yields on government bonds, which are especially attractive for retirement savings because they are such a safe investment.",
                    "m_m": "A lot of people are making large monthly interest payments on credit card debt, and they have no money left to save for retirement.",
                },
            },
            "Y": {
                "name": "trade deficits",
                "detailed": "A country's trade deficit is the difference between the value of the goods that a country imports and the value of the goods that a country exports.",
                "p_value": {"1": "small", "0": "normal"},
                "m_value": {"1": "large", "0": "normal"},
                "explanations": {
                    "p_p": "When the economy is good, people can cover their basic expenses and so have enough money left over to contribute to their retirement accounts.",
                    "p_m": "When the economy is good, people are optimistic and so spend rather than save.",
                    "m_p": "People become nervous when their economy is no longer competitive enough in the world economy to export products, and begin saving for retirement as a result.",
                    "m_m": "The loss of local manufacturing jobs means that there are people out of work, and contributions to retirement accounts decreases.",
                },
            },
            "Z": {
                "name": "retirement savings",
                "detailed": "Retirement savings is the money people save for their retirement.",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
            },
        },
        "introduction": "Economists seek to describe and predict the regular patterns of economic fluctuation. To do this, they study some important variables or attributes of economies. They also study how these attributes are responsible for producing or causing one another.",
    },
    "sociology": {
        "domain_name": "sociology",
        "variables": {
            "X": {
                "name": "urbanization",
                "detailed": "Urbanization is the degree to which the members of a society live in urban environments (i.e., cities) versus rural environments.",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
                "explanations": {
                    "p_p": "Big cities provide many opportunities for financial and social improvement.",
                    "p_m": "In big cities many people are competing for the same high-status jobs and occupations.",
                    "m_p": 'People in rural areas are rarely career oriented, and so take time off from working and switch frequently between different "temp" jobs.',
                    "m_m": "The low density of people prevents the dynamic economic expansion needed for people to get ahead.",
                },
            },
            "Y": {
                "name": "interest in religion",
                "detailed": "Interest in religion is the degree to which the members of a society show a curiosity in religion issues or participate in organized religions.",
                "p_value": {"1": "low", "0": "normal"},
                "m_value": {"1": "high", "0": "normal"},
                "explanations": {
                    "p_p": "Without the restraint of religion-based morality, the impulse toward greed dominates and people tend to accumulate material wealth.",
                    "p_m": "Many religions reinforce a strong work ethic; without this motivation, workers become complacent at their jobs.",
                    "m_p": "Religion fosters communal care, and those of the same religion tend to support each other with jobs, financial favors, and so on.",
                    "m_m": "The spiritualism induced by religion works to reduce the desire for material wealth.",
                },
            },
            "Z": {
                "name": "socio-economic mobility",
                "detailed": "Socioeconomic mobility is the degree to which the members of a society are able to improve their social and economic status.",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
            },
        },
        "introduction": "Sociologists seek to describe and predict the regular patterns of societal interactions. To do this, they study some important variables or attributes of societies. They also study how these attributes are responsible for producing or causing one another.",
    },
    "weather": {
        "domain_name": "weather",
        "variables": {
            "X": {
                "name": "ozone levels",
                "detailed": "Ozone is a gaseous allotrope of oxygen (O3) and is formed by exposure to UV radiation.",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
                # Note: this is graph dependent. the explanations below are based on the collider graph
                "explanations": {
                    "p_m": "Ozone attracts extra oxygen atoms from water molecules, creating a concentration of water vapor in that region.",
                    # "p_p": "Ozone accepts extra oxygen atoms, decreasing the amount of oxygen available to form water molecules. With fewer water molecules, there is lower humidity.",
                    "p_p": "Ozone tends to rapidly break down into normal oxygen, leaving more oxygen available for the formation of water molecules (H2O).",
                    "m_p": "The oxygen atoms that would normally be part of ozone molecules are free to combine with hydrogen atoms instead, creating water molecules.",
                    "m_m": "The low amount of ozone allows a large number of ultra-violet (UV) rays to enter the atmosphere, and the UV rays break up water molecules, resulting in low humidity.",
                },
            },
            "Y": {
                "name": "air pressure",
                "detailed": "Air pressure is force exerted due to concentrations of air molecules.",
                "p_value": {"1": "low", "0": "normal"},
                "m_value": {"1": "high", "0": "normal"},
                # Note: this is graph dependent. the explanations below are based on the collider graph
                "explanations": {
                    "p_p": "When pressure does not force water vapor to break into oxygen and hydrogen atoms, water vapor remains in abundance.",
                    "p_m": "Low air pressure poorly facilitates condensation; as a result, there are less water molecules in the air.",
                    "m_p": "The higher pressure means that the components of water molecules (hydrogen and oxygen) tend to not dissocciate from one another. Because there are more water molecules, humidity is higher.",
                    "m_m": "When air pressure is high, water vapor condenses into liquid water (rain), and the atmosphere is left with little moisture.",
                },
            },
            "Z": {
                "name": "humidity",
                "detailed": "Humidity is the degree to which the atmosphere contains water molecules.",
                "p_value": {"1": "high", "0": "normal"},
                "m_value": {"1": "low", "0": "normal"},
            },
        },
        "introduction": "Meteorologists seek to describe and predict the regular patterns that govern weather systems. To do this, they study some important variables or attributes of weather systems. They also study how these attributes are responsible for producing or causing one another.",
    },
}


#  graph structures (currently only collider in dataset, but expandable), RW17 used collider and fork.
graph_structures = {
    "collider": {
        "description": "X→Z←Y",  # RW17: A→C←B
        "causal_template": "{x_sense} {x_name} causes {z_sense} {z_name}. Also, {y_sense} {y_name} causes {z_sense} {z_name}.",
    },
    "fork": {
        "description": "B←A→C",  # RW17: C←A→B
        "causal_template": "{x_sense} {x_name} causes {y_sense} {y_name}. Also, {x_sense} {x_name} causes {z_sense} {z_name}.",
    },
    "chain": {
        "description": "A→B→C",  # RW17: A→B→C
        "causal_template": "{x_sense} {x_name} causes {y_sense} {y_name}. And {y_sense} {y_name} causes {z_sense} {z_name}.",
    },
}


# Add inference task mapping to the structure
inference_tasks_rw17 = {
    "a": {
        "query_node": "Xi=1",
        "observation": "Z=1, Yj=1",
        "query": "p(Xi=1|Z=1, Yj=1)",
    },
    "b": {"query_node": "Xi=1", "observation": "Z=1", "query": "p(Xi=1|Z=1)"},
    "c": {
        "query_node": "Xi=1",
        "observation": "Z=1, Yj=0",
        "query": "p(Xi=1|Z=1, Yj=0)",
    },
    "d": {"query_node": "Xi=1", "observation": "Yj=1", "query": "p(Xi=1|Yj=1)"},
    "e": {"query_node": "Xi=1", "observation": "Yj=0", "query": "p(Xi=1|Yj=0)"},
    "f": {
        "query_node": "Xi=1",
        "observation": "Z=0, Yj=1",
        "query": "p(Xi=1|Z=0, Yj=1)",
    },
    "g": {"query_node": "Xi=1", "observation": "Z=0", "query": "p(Xi=1|Z=0)"},
    "h": {
        "query_node": "Xi=1",
        "observation": "Z=0, Yj=0",
        "query": "p(Xi=1|Z=0, Yj=0)",
    },
    "i": {
        "query_node": "Z=1",
        "observation": "Xi=0, Yj=0",
        "query": "p(Z=1|Xi=0, Yj=0)",
    },
    "j": {
        "query_node": "Z=1",
        "observation": "Xi=0, Yj=1",
        "query": "p(Z=1|Xi=0, Yj=1)",
    },
    "k": {
        "query_node": "Z=1",
        "observation": "Xi=1, Yj=1",
        "query": "p(Z=1|Xi=1, Yj=1)",
    },
}
