import pandas as pd


def validate_season(pipeline, X, y, is_rookie: bool) -> tuple[pd.DataFrame, int]:
    assert (
        X["YEAR"].nunique() == 1
    ), f"Expected data for a single season, but got {X['YEAR'].nunique()} seasons"
    assert "PLAYER_NAME" in X.columns
    assert len(X) == len(y), "Mismatch in number of samples between X and y"
    assert X[
        "PLAYER_NAME"
    ].is_unique, f"Expected unique player names, but found duplicates: {X['PLAYER_NAME'][X['PLAYER_NAME'].duplicated()]}"

    predicted_vote_shares = pipeline.predict(X)

    df_res = pd.DataFrame(
        {
            "PLAYER_NAME": X["PLAYER_NAME"],
            "predicted_vote_share": predicted_vote_shares,
            "actual_vote_share": y,
        }
    )

    df_res["predicted_place"] = (
        df_res["predicted_vote_share"]
        .rank(ascending=False, method="first")
        .apply(lambda r: assign_team(r, is_rookie=is_rookie))
    )
    df_res["actual_place"] = (
        df_res["actual_vote_share"]
        .rank(ascending=False, method="first")
        .apply(lambda r: assign_team(r, is_rookie=is_rookie))
    )

    score = calculate_score(df_res["predicted_place"], df_res["actual_place"])

    return df_res, score


def calculate_score(predicted_place: pd.Series, actual_place: pd.Series) -> int:
    df = pd.DataFrame({"pred": predicted_place, "act": actual_place})

    assert not df["pred"].isna().any()

    total_score = 0
    bonus_map = {1: 0, 2: 5, 3: 10, 4: 20, 5: 40}

    for team_val, group in df.groupby("pred"):
        exact_matches = (group["act"] != 1000) & (group["act"] == group["pred"])
        off_by_1 = (group["act"] != 1000) & (abs(group["act"] - group["pred"]) == 1)
        off_by_2 = (group["act"] != 1000) & (abs(group["act"] - group["pred"]) == 2)

        team_score = (
            (exact_matches.sum() * 10) + (off_by_1.sum() * 8) + (off_by_2.sum() * 6)
        )
        total_score += team_score

        correct_count = exact_matches.sum()

        if correct_count >= 5:
            total_score += 40
        elif correct_count in bonus_map:
            total_score += bonus_map[correct_count]

    return int(total_score)


def assign_team(rank, is_rookie: bool):
    if rank <= 5:
        return 1

    if rank <= 10:
        return 2

    if rank <= 15 and not is_rookie:
        return 3

    return 1000
