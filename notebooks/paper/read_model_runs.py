import pandas as pd
import os
import glob
import re

import pandas as pd

def load_janderson(path=""):
    exam_df = pd.read_csv(path)
    return exam_df

def get_question_wide_df(question_long_df, firac_order):
    """
    Converte question_long_df para formato wide:

    - cada linha: (question_id, model_name)
    - cada coluna: um valor de FIRAC (ordenadas por firac_order)
    - valores: is_correct
    """

    question_wide_df = (
        question_long_df
        .pivot_table(
            index=["question_id", "model_name"],
            columns="firac",
            values="is_correct",
            aggfunc="first"   # seguro se já houver unicidade
        )
        .reset_index()
    )

    # Remove o nome do eixo de colunas
    question_wide_df.columns.name = None

    # ----------------------------
    # Reordena colunas FIRAC
    # ----------------------------
    base_cols = ["question_id", "model_name"]
    firac_cols = [c for c in firac_order if c in question_wide_df.columns]

    question_wide_df = question_wide_df[
        base_cols + firac_cols
    ]

    return question_wide_df



def read_model_runs(base_folder='../../data/processed/model-runs', exam_path="../../data/processed/oab_with_firac_portuguese_shuffle.csv"):

    exam_df = pd.read_csv(exam_path)

    question_results = []


    # Lista todas as pastas dentro de model-runs (cada pasta é um idioma)
    languages = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]
    languages = ["portuguese"]

    for language in languages:
        folder_path = os.path.join(base_folder, language)
        csv_files = glob.glob(os.path.join(folder_path, '*_answers_firac*.csv'))

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            match = re.match(r'(.+)_answers_firac-(.*)\.csv$', filename)
            if match:
                model_name = match.group(1)
                firac = match.group(2)
            else:
                # Pula arquivos que não seguem o padrão
                continue

            # Lê o CSV
            df = pd.read_csv(filepath)

            # Adiciona colunas extras
            df["model_name"] = model_name
            df["firac"] = firac
            df["language"] = language  # adiciona o idioma

            question_results.append(df)

    # Concatena todos os DataFrames
    final_df = pd.concat(question_results, ignore_index=True)

    # JANDERSON
    question_janderson_df = load_janderson("../../data/processed/model-runs/portuguese/FULL/qwen_oab.csv")
    question_janderson_df = question_janderson_df.rename(columns={"model": "model_name"})
    question_janderson_df["is_correct"] = (question_janderson_df["answer"] == question_janderson_df["correct_option"])
    question_janderson_df = question_janderson_df.assign(
        firac=question_janderson_df["hint"].map({
            "no_hint": "_____",
            "fact": "F____",
            "issue": "FI___",
            "rule": "FIR__",
            "application": "FILA_",
            "conclusion": "FIRAC"
        })
    )
    question_janderson_df = pd.concat(
    [   question_janderson_df,
        question_janderson_df[question_janderson_df.firac == "_____"].assign(firac="unstructured")],
        ignore_index=True
    )


    final_df = pd.concat(
    [final_df, question_janderson_df],
    axis=0,
    ignore_index=True,
    sort=False
    )


    # Reordena colunas
    cols = ["model_name", "firac", "language", "is_correct"] + [c for c in final_df.columns if c not in ["model_name", "firac", "language", "is_correct"]]
    final_df = final_df[cols]

    final_df = final_df.merge(
        exam_df[["question_id", "tema"]],
        on="question_id",
        how="left"
    )


    return final_df

def filter_complete_questions(question_long_df, models, firacs):
    """
    Retorna:
    - filtered_df: apenas questões completas
    - question_wide_df
    - model_order
    - firac_order
    - completeness_map_df: % de cobertura por (model_name, firac),
      com zeros explícitos para combinações ausentes
    """

    # ----------------------------
    # 1. Filtro inicial
    # ----------------------------
    df = (
        question_long_df
        .loc[
            question_long_df["model_name"].isin(models)
            & question_long_df["firac"].isin(firacs)
        ]
        .copy()
    )

    total_questions = df["question_id"].nunique()

    # ----------------------------
    # 2. COMPLETENESS MAP (com grid completo)
    # ----------------------------
    # grid esperado (modelo × firac)
    expected_grid = (
        pd.MultiIndex
        .from_product([models, firacs], names=["model_name", "firac"])
        .to_frame(index=False)
    )

    observed_coverage = (
        df
        .drop_duplicates(subset=["question_id", "model_name", "firac"])
        .groupby(["model_name", "firac"])["question_id"]
        .nunique()
        .reset_index(name="n_questions")
    )

    completeness_map_df = (
        expected_grid
        .merge(
            observed_coverage,
            on=["model_name", "firac"],
            how="left"
        )
    )

    completeness_map_df["n_questions"] = (
        completeness_map_df["n_questions"].fillna(0).astype(int)
    )

    completeness_map_df["completeness_pct"] = (
        completeness_map_df["n_questions"] / total_questions
        if total_questions > 0 else 0
    )

    # ----------------------------
    # 3. Combinações ESPERADAS
    # ----------------------------
    total_combinations = len(models) * len(firacs)

    # ----------------------------
    # 4. Cobertura por question_id
    # ----------------------------
    question_coverage = (
        df
        .groupby("question_id")[["model_name", "firac"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name="n_combinations")
    )

    # ----------------------------
    # 5. Question_ids completos
    # ----------------------------
    complete_question_ids = question_coverage.loc[
        question_coverage["n_combinations"] == total_combinations,
        "question_id"
    ]

    filtered_df = (
        df
        .loc[df["question_id"].isin(complete_question_ids)]
        .copy()
    )

    # ----------------------------
    # 6. Ordem dos MODELOS por acurácia
    # ----------------------------
    model_order = (
        filtered_df
        .groupby("model_name")["is_correct"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
        if not filtered_df.empty else []
    )

    # ----------------------------
    # 7. Ordem dos FIRACs por acurácia
    # ----------------------------
    firac_order = (
        filtered_df
        .groupby("firac")["is_correct"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
        if not filtered_df.empty else []
    )

    return (
        filtered_df,
        get_question_wide_df(filtered_df, firac_order),
        model_order,
        firac_order,
        completeness_map_df
    )
