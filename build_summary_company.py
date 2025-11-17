import llm
import os
import pandas as pd
import re

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert assistant in sustainability and GRI standards. "
            "Your task is to analyze data extracted from a company's PDFs in the form of CSV tables "
            "related to specific GRI indicators, and provide a clear, concise summary of the company's performance. "
            "Instructions: "
            "- Base your summary strictly on the data provided in the CSV tables. "
            "- Highlight trends, improvements, or regressions in the company's performance where possible. "
            "- Do not add assumptions or information not present in the tables. "
            "- For each key point, reference the row, cell, page, and table number used from the CSV context. "
            "- Make the summary concise, well-structured, and readable for stakeholders."
            "- if there is no context, reply clearly that you have not received any information. Nothing else. "
        )
    },
    {
        "role": "user",
        "content": (
            "Here are the CSV tables extracted from the company's PDFs related to GRI indicators:\n---\n{context}\n---\n"
            "Please provide a concise summary of the company's performance based strictly on this data. "
        )
    }
]


def build_summary(company):
    folder = os.path.join("table_dataset", company)

    csvs = [f for f in os.listdir(folder) if f.endswith(".csv")]
    csv_texts = []

    for csv in csvs:

        csv_file = os.path.join("table_dataset", company, csv)
        df = pd.read_csv(csv_file, sep=';')
        page = int(re.search(r"(\d+)_\d+\.csv$", csv_file).group(1))
        num = int(re.search(r"_(\d+)\.csv$", csv_file).group(1))
        csv_texts.append(f"# Page {page}, Table {num}\n")
        csv_texts.append(df.to_csv(index=False))

    tables_str = "\n\n".join(csv_texts)
    header = f"Company: {company}\n"
    context = f"{header}\n\n{tables_str}\n---\n"

    message = [
        messages[0],
        {
            "role": "user",
            "content": messages[1]["content"].format(
                context=context
            )
        }
    ]

    response = llm.ask_openai(message)
    output_file = os.path.join(folder, "summary.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    return response

