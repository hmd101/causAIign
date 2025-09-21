# src/causalign/prompts/core/utils.py

import random
import string
from typing import List, Sequence

import pandas as pd


# Append the dataframes together
def append_dfs(*dfs):
    """
    Append multiple dataframes together after checking that they have the same columns and dtypes.

    Parameters:
    -----------
    *dfs : list of pd.DataFrame
        DataFrames to be appended.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame resulting from appending the input DataFrames.
    """
    # Check that all dataframes have the same columns and dtypes
    columns = dfs[0].columns
    dtypes = dfs[0].dtypes

    for df in dfs[1:]:
        assert df.columns.equals(columns), "DataFrames have different columns"
        assert df.dtypes.equals(dtypes), "DataFrames have different dtypes"

    # Concatenate the dataframes
    all_domains_df = pd.concat(dfs, ignore_index=True)
    return all_domains_df


# -----------------------------------------------
# Random string helpers for abstract domain names
# -----------------------------------------------

# Predefined character pools
CHAR_POOLS = {
    "lower": string.ascii_lowercase,
    "upper": string.ascii_uppercase,
    "letters": string.ascii_letters,
    "digits": string.digits,
    "symbols": string.punctuation,  # e.g., !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    "alnum": string.ascii_letters + string.digits,
    "letters_symbols": string.ascii_letters + string.punctuation,
    "digits_symbols": string.digits + string.punctuation,
    "all": string.ascii_letters + string.digits + string.punctuation,
}


def random_string(
    length: int,
    *,
    chars: str | Sequence[str] = "letters",
    seed: int | None = None,
) -> str:
    """
    Generate a random string of given length from a specified character pool.

    Args:
        length: Number of characters in the output string.
        chars: Either a pool name from CHAR_POOLS (e.g., "letters", "digits",
               "symbols", "alnum", "all") or an explicit iterable/string of
               characters to sample from.
        seed: Optional seed for deterministic output.

    Returns:
        Randomly generated string.
    """
    if isinstance(chars, str) and chars in CHAR_POOLS:
        pool = CHAR_POOLS[chars]
    else:
        # Treat provided chars as the literal pool
        pool = "".join(chars)  # supports str or any iterable of single-char strings
    if not pool:
        raise ValueError("Character pool is empty")

    rng = random.Random(seed) if seed is not None else random
    return "".join(rng.choice(pool) for _ in range(length))


def random_variable_names(
    count: int,
    length: int,
    *,
    chars: str | Sequence[str] = "letters",
    unique: bool = False,
    seed: int | None = None,
    max_attempts: int = 10000,
) -> List[str]:
    """
    Generate multiple random names for variables.

    Args:
        count: How many names to create.
        length: Length of each name.
        chars: Pool name (see CHAR_POOLS) or explicit character sequence.
        unique: If True, ensure all names are unique (best-effort within max_attempts).
        seed: Optional seed for reproducibility; applied to an internal RNG.
        max_attempts: Upper bound on attempts when enforcing uniqueness.

    Returns:
        List of names.
    """
    rng = random.Random(seed) if seed is not None else random

    # Resolve pool once for efficiency
    if isinstance(chars, str) and chars in CHAR_POOLS:
        pool = CHAR_POOLS[chars]
    else:
        pool = "".join(chars)
    if not pool:
        raise ValueError("Character pool is empty")

    def _one():
        return "".join(rng.choice(pool) for _ in range(length))

    if not unique:
        return [_one() for _ in range(count)]

    # Enforce uniqueness
    names = set()
    attempts = 0
    while len(names) < count and attempts < max_attempts:
        names.add(_one())
        attempts += 1
    if len(names) < count:
        raise RuntimeError(
            f"Could not generate {count} unique names of length {length} from given pool after {attempts} attempts"
        )
    return list(names)
