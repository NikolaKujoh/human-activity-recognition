import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE = "UCI HAR Dataset"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
OUTPUT_DIR = "data"

ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def get_unique_feature_names(features_path):
    """Ucitaj nazive feature-a i resi duplikate dodavanjem sufiksa n/nn."""
    with open(features_path, encoding="utf-8") as f:
        features = [line.split()[1] for line in f.readlines()]

    seen = set()
    uniq_features = []
    for x in features:
        if x not in seen:
            uniq_features.append(x)
            seen.add(x)
        elif x + "n" not in seen:
            uniq_features.append(x + "n")
            seen.add(x + "n")
        else:
            uniq_features.append(x + "nn")
            seen.add(x + "nn")
    return uniq_features


def load_split(split_dir, feature_names, subject_file, y_file):
    """Ucitaj X, subject i y za train/test i vrati spojeni dataframe."""
    x_path = os.path.join(split_dir, "X_" + split_dir.split(os.sep)[-1] + ".txt")
    X = pd.read_csv(x_path, sep=r"\s+", header=None, names=feature_names)

    subject = pd.read_csv(os.path.join(split_dir, subject_file), header=None).squeeze("columns")
    y = pd.read_csv(os.path.join(split_dir, y_file), names=["Activity"]).squeeze("columns")
    y_labels = y.map(ACTIVITY_MAP)

    df = X.copy()
    df["subject"] = subject
    df["Activity"] = y
    df["ActivityName"] = y_labels
    return df


def clean_column_names(columns):
    """Ukloni specijalne karaktere iz naziva kolona."""
    cleaned = columns.str.replace(r"[()]", "", regex=True)
    cleaned = cleaned.str.replace(r"[-]", "", regex=True)
    cleaned = cleaned.str.replace(r"[,]", "", regex=True)
    return cleaned


def main():
    print("# UCI - Dataset - Preprocessing")

    feature_names = get_unique_feature_names(os.path.join(BASE, "features.txt"))
    print(f"No of Features: {len(feature_names)}")

    train = load_split(TRAIN_DIR, feature_names, "subject_train.txt", "y_train.txt")
    test = load_split(TEST_DIR, feature_names, "subject_test.txt", "y_test.txt")

    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")

    print(f"No of duplicates in train: {int(train.duplicated().sum())}")
    print(f"No of duplicates in test : {int(test.duplicated().sum())}")

    print(f"We have {int(train.isnull().values.sum())} NaN/Null values in train")
    print(f"We have {int(test.isnull().values.sum())} NaN/Null values in test")

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    plt.figure(figsize=(16, 8))
    plt.title("Data provided by each user", fontsize=20)
    sns.countplot(x="subject", hue="ActivityName", data=train)
    plt.tight_layout()
    plt.savefig("01_subject_activity_count.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("No of Datapoints per Activity", fontsize=15)
    sns.countplot(x="ActivityName", data=train)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("02_activity_count.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Bar-plot: srednje vrednosti po klasi + error bar (±std)
    # Podeljeno u figure po 3 prediktora
    # ------------------------------------------------------------------
    REP_PREFIXES = [
        ("tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z"),
        ("tGravityAcc-mean()-X", "tGravityAcc-mean()-Y", "tGravityAcc-mean()-Z"),
        ("tBodyAccJerk-mean()-X", "tBodyAccJerk-mean()-Y", "tBodyAccJerk-mean()-Z"),
        ("tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z"),
        ("tBodyAccMag-mean()", "tBodyGyroMag-mean()", "fBodyAcc-mean()-X"),
    ]
    rep_features = [f for group in REP_PREFIXES for f in group if f in feature_names]
    rep_features = list(dict.fromkeys(rep_features))  # deduplicate, preserve order

    CHUNK = 3
    chunks = [rep_features[i:i + CHUNK] for i in range(0, len(rep_features), CHUNK)]

    for fig_idx, chunk in enumerate(chunks, start=1):
        plot_df = train[chunk + ["ActivityName"]].copy()
        df_melt = plot_df.melt(id_vars="ActivityName", var_name="Feature", value_name="Value")

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(
            data=df_melt,
            x="Feature", y="Value", hue="ActivityName",
            estimator="mean", errorbar="sd",
            palette="tab10", ax=ax,
            capsize=0.08
        )
        ax.set_title(
            f"Srednje vrednosti prediktora po klasi (±std)  —  grupa {fig_idx}/{len(chunks)}",
            fontsize=13
        )
        ax.set_xlabel("")
        ax.set_ylabel("Mean ± Std")
        ax.legend(title="Aktivnost", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(f"03_mean_std_barplot_group{fig_idx}.png", dpi=120)
        plt.close()

        # Violin plot za istu grupu prediktora
        plt.figure(figsize=(14, 8))
        sns.violinplot(
            data=df_melt,
            x="Feature",
            y="Value",
            hue="ActivityName",
            inner="quartile",
            palette="muted",
            bw_adjust=0.5,
            cut=0,
        )
        plt.title(f"Violin plot distribucija — grupa {fig_idx}")
        plt.xticks(rotation=45)
        plt.legend(title="Aktivnost", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"03_violin_group{fig_idx}.png", dpi=120)
        plt.close()

        # Boxplot (brza provera outlier-a) za istu grupu
        plt.figure(figsize=(14, 7))
        sns.boxplot(
            data=df_melt,
            x="Feature",
            y="Value",
            hue="ActivityName",
            palette="tab10",
            fliersize=1,
            linewidth=0.8,
        )
        plt.title(f"Boxplot distribucija i outlier-i — grupa {fig_idx}")
        plt.xticks(rotation=45)
        plt.legend(title="Aktivnost", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"03_boxplot_group{fig_idx}.png", dpi=120)
        plt.close()

    # ------------------------------------------------------------------
    # Histogrami primera po klasama za svaki prediktor
    # Seaborn displot format, podeljeno u figure sa istim brojem prediktora
    # ------------------------------------------------------------------
    PER_FIG = 3
    hist_chunks = [rep_features[i:i + PER_FIG] for i in range(0, len(rep_features), PER_FIG)]

    for fig_idx, chunk in enumerate(hist_chunks, start=1):
        plot_df = train[chunk + ["ActivityName"]].copy()
        df_melt = plot_df.melt(id_vars="ActivityName", var_name="Feature", value_name="Value")

        g = sns.displot(
            data=df_melt,
            x="Value",
            col="Feature",
            col_wrap=3,
            hue="ActivityName",
            kind="hist",
            bins=30,
            kde=True,
            stat="count",
            common_norm=False,
            common_bins=False,
            element="step",
            fill=False,
            height=3.2,
            aspect=1.2,
            facet_kws=dict(margin_titles=True),
        )
        g.set_axis_labels("Vrednost", "Frekvencija")
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.86)
        g.fig.suptitle(
            f"Histogrami prediktora po klasama — grupa {fig_idx}/{len(hist_chunks)}",
            fontsize=13,
        )
        g.savefig(f"04_histograms_group{fig_idx}.png", dpi=120)
        plt.close(g.fig)

    # ------------------------------------------------------------------
    # Korelacije: prediktor-prediktor i prediktor-ciljna promenljiva
    # ------------------------------------------------------------------
    print("\n# Korelacije")

    # 1) Korelacije izmedju reprezentativnih prediktora (vizuelno)
    corr_rep = train[rep_features].corr(method="pearson")
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_rep, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Korelaciona matrica reprezentativnih prediktora")
    plt.tight_layout()
    plt.savefig("05_corr_matrix_representative.png", dpi=120)
    plt.close()

    # 2) Najjace korelacije izmedju svih parova prediktora
    corr_all_abs = train[feature_names].corr(method="pearson").abs()
    upper_mask = np.triu(np.ones(corr_all_abs.shape), k=1).astype(bool)
    pair_corr = corr_all_abs.where(upper_mask).stack().sort_values(ascending=False)
    top_pairs = pair_corr.head(20).reset_index()
    top_pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    top_pairs.to_csv("05_top_predictor_pairs_corr.csv", index=False)
    print("Top 10 parova prediktora po apsolutnoj korelaciji:")
    print(top_pairs.head(10).to_string(index=False))

    # 3) Korelacija prediktora sa ciljnom promenljivom preko one-hot klasa
    y_onehot = pd.get_dummies(train["ActivityName"], prefix="cls")
    corr_target_oh = train[feature_names].corrwith(y_onehot.iloc[:, 0], method="pearson")
    corr_target_oh_df = pd.DataFrame({"feature": feature_names})
    for cls_col in y_onehot.columns:
        corr_target_oh_df[f"corr_{cls_col}"] = train[feature_names].corrwith(y_onehot[cls_col], method="pearson").values
    corr_target_oh_df["max_abs_corr_any_class"] = corr_target_oh_df[[c for c in corr_target_oh_df.columns if c.startswith("corr_cls_")]].abs().max(axis=1)
    corr_target_oh_df = corr_target_oh_df.sort_values("max_abs_corr_any_class", ascending=False)
    corr_target_oh_df.to_csv("06_predictor_target_corr_onehot.csv", index=False)
    print("\nTop 10 prediktora po max |korelaciji| sa nekom klasom (one-hot):")
    print(corr_target_oh_df[["feature", "max_abs_corr_any_class"]].head(10).to_string(index=False))

    # 4) Korelacija prediktora sa static vs dynamic metom
    static_set = {"SITTING", "STANDING", "LAYING"}
    is_static = train["ActivityName"].isin(static_set).astype(int)
    corr_static = train[feature_names].corrwith(is_static, method="pearson")
    corr_static_df = corr_static.sort_values(key=np.abs, ascending=False).reset_index()
    corr_static_df.columns = ["feature", "corr_with_is_static"]
    corr_static_df.to_csv("07_predictor_static_dynamic_corr.csv", index=False)
    print("\nTop 10 prediktora po apsolutnoj korelaciji sa static-vs-dynamic:")
    print(corr_static_df.head(10).to_string(index=False))

    # Bar-plot najboljih 20 prediktora prema static-vs-dynamic meti
    top_target = corr_static_df.head(20)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_target, y="feature", x="corr_with_is_static", hue="feature", palette="vlag", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 20 korelacija prediktor sa static-vs-dynamic")
    plt.xlabel("Pearson korelacija sa IsStatic")
    plt.ylabel("Prediktor")
    plt.tight_layout()
    plt.savefig("07_predictor_static_dynamic_top20.png", dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Stacionarnost (subject 1): promena nivoa i varijanse kroz indeks
    # ------------------------------------------------------------------
    subject_col = "subject"
    series_feature = "tBodyAcc-mean()-X"
    if series_feature in train.columns:
        sub1 = train[train[subject_col] == 1].reset_index(drop=True)
        if len(sub1) > 0:
            s = sub1[series_feature]
            rolling_window = 128
            roll_mean = s.rolling(rolling_window, min_periods=1).mean()
            roll_std = s.rolling(rolling_window, min_periods=1).std()

            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            sns.lineplot(x=np.arange(len(s)), y=s.values, ax=axes[0], linewidth=1)
            sns.lineplot(x=np.arange(len(roll_mean)), y=roll_mean.values, ax=axes[0], color="tomato", linewidth=1.2)
            axes[0].set_title("Subject 1: tBodyAcc-mean()-X kroz vreme (sa rolling mean)")
            axes[0].set_ylabel("Vrednost")

            sns.lineplot(x=np.arange(len(roll_std)), y=roll_std.values, ax=axes[1], color="darkorange", linewidth=1.2)
            axes[1].set_title("Rolling std (window=128)")
            axes[1].set_xlabel("Indeks prozora")
            axes[1].set_ylabel("Std")

            plt.tight_layout()
            plt.savefig("08_stationarity_subject1_tBodyAcc_mean_X.png", dpi=120)
            plt.close()

    cleaned_columns = clean_column_names(train.columns)
    train.columns = cleaned_columns
    test.columns = cleaned_columns

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved: data/train.csv")
    print("Saved: data/test.csv")


if __name__ == "__main__":
    main()

