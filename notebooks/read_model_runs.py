import pandas as pd
import os
import glob
import re

import pandas as pd

def gera_question_wide_df(question_long_df, firac_order=None):
    """
    Constrói um DataFrame wide em que:
    - linhas: question_id
    - colunas: valores de firac
    - valores: acurácia média (mean of is_correct)
    - firac_order: lista opcional para ordenar as colunas
    """
    # Agrupa por question_id e firac, calculando acurácia média
    question_acc = (
        question_long_df.groupby(['question_id', 'firac'])['is_correct']
        .mean()
        .reset_index()
    )
    
    # Pivot para wide format
    question_wide_df = question_acc.pivot(index='question_id', columns='firac', values='is_correct')
    
    # Reordena colunas se firac_order for fornecido
    if firac_order is not None:
        firac_order_filtered = [f for f in firac_order if f in question_wide_df.columns]
        question_wide_df = question_wide_df[firac_order_filtered]
    
    # Ordena as linhas pelo valor da coluna firac "____"
    if '____' in question_wide_df.columns:
        question_wide_df = question_wide_df.sort_values(by='____', ascending=False)
    
    return question_wide_df



def gera_model_wide_df(question_long_df, firac_order=None):
    """
    Constrói um DataFrame wide em que:
    - linhas: model_name
    - colunas: valores de firac
    - valores: acurácia média (mean of is_correct)
    - firac_order: lista opcional para ordenar as colunas
    """
    # Agrupa por modelo e firac, calculando acurácia média
    model_acc = (
        question_long_df.groupby(['model_name', 'firac'])['is_correct']
        .mean()
        .reset_index()
    )
    
    # Pivot para wide format
    model_wide_df = model_acc.pivot(index='model_name', columns='firac', values='is_correct')
    
    # Reordena colunas se firac_order for fornecido
    if firac_order is not None:
        # Mantém apenas os firac que existem no DataFrame
        firac_order_filtered = [f for f in firac_order if f in model_wide_df.columns]
        model_wide_df = model_wide_df[firac_order_filtered]
    
    return model_wide_df



def read_model_runs(base_folder='../data/processed/model-runs'):
    question_results = []

    # Lista todas as pastas dentro de model-runs (cada pasta é um idioma)
    languages = [name for name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, name))]

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

    # --- Calcula firac_order ---
    firac_order = (
        final_df.groupby('firac')['is_correct']
        .mean()
        .sort_values(ascending=False)  # maior acurácia primeiro
        .index
        .tolist()
    )

    model_name_order = (
        final_df.groupby('model_name')['is_correct']
        .mean()
        .sort_values(ascending=False)  # maior acurácia primeiro
        .index
        .tolist()
    )


    return final_df, gera_model_wide_df(final_df, firac_order), gera_question_wide_df(final_df, firac_order), firac_order, model_name_order
