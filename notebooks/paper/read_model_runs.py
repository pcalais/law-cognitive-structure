import pandas as pd
import os
import glob
import re

import pandas as pd

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
    1) Filtra o DataFrame pelos models e firacs desejados
    2) Mantém apenas question_id presentes em TODAS as combinações
       (model_name, firac) desse subconjunto
    3) Retorna também:
       - model_order: modelos ordenados por acurácia média (↑)
       - firac_order: firacs ordenados por acurácia média (↑)
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

    # ----------------------------
    # 2. Combinações esperadas
    # ----------------------------
    total_combinations = (
        df[["model_name", "firac"]]
        .drop_duplicates()
        .shape[0]
    )

    # ----------------------------
    # 3. Cobertura por question_id
    # ----------------------------
    question_coverage = (
        df
        .groupby("question_id")[["model_name", "firac"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .reset_index(name="n_combinations")
    )

    # ----------------------------
    # 4. Question_ids completos
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
    # 5. Ordem dos MODELOS por acurácia
    # ----------------------------
    model_order = (
        filtered_df
        .groupby("model_name")["is_correct"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    # ----------------------------
    # 6. Ordem dos FIRACs por acurácia
    # ----------------------------
    firac_order = (
        filtered_df
        .groupby("firac")["is_correct"]
        .mean()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    return filtered_df, get_question_wide_df(filtered_df, firac_order), model_order, firac_order
