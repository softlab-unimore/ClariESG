import pandas as pd
from prompts.query_agent_prompts import prompt_extract, prompt_fallback_python, prompt_normalization, prompt_total
from llm import ask_openai
from typing import Union, List, Dict
import io
import re
import contextlib


class QueryAgent:
    def __init__(self):
        pass

    def remove_markdown_syntax(self, text: str) -> str:
        # Remove triple backtick code blocks (```python ... ```)
        # text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", '', m.group()), text)

        # Remove inline code (`code`)
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Remove bold (**text** or __text__)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Remove italic (*text* or _text_)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        # Remove blockquotes
        text = re.sub(r"^>\s?", '', text, flags=re.MULTILINE)

        # text = text.replace("python", "")
        return text.strip()

    def extract_result(self, text: str, pattern: str) -> str:
        position = text.lower().rfind(pattern.lower())
        if position == -1:
            print(f"Cannot find pattern '{pattern}' in '{text}'. Defaulting to '{text}'...")
            return text
        else:
            position += len(pattern)
        return text[position:].strip()

    def filter_table(self, query: str, table: pd.DataFrame) -> Union[tuple[pd.DataFrame, str], tuple[int, str]]:
        # add row index
        try:
            table = table.drop(columns="index")
        except:
            pass

        table.insert(0, "index", range(len(table)))

        # add column index
        table = pd.concat([pd.DataFrame([table.columns.tolist()], columns=table.columns), table], ignore_index=True)
        table.columns = range(len(table.columns))

        prompt_extract_filled = prompt_extract.format(question=query, table=table.to_html(index=False))
        response = ask_openai([
            {
                "role": "system",
                "content": prompt_extract_filled,
            }
        ])

        rows_columns_extracted = self.remove_markdown_syntax(self.extract_result(response, "Final answer:"))

        try:
            rows_columns_extracted = eval(rows_columns_extracted)
        except:
            print(f"Formatting error while extracting the row and column indices: '{rows_columns_extracted}'")
            return -1, response

        table.columns = table.columns.astype(str)
        columns = [el for i, el in enumerate(table.iloc[0, :]) if i in rows_columns_extracted["columns"]]

        table = table[table["0"].isin(rows_columns_extracted["rows"])]  # table["0"] is "index"
        table = table[[str(el) for el in rows_columns_extracted["columns"]]]

        # clean
        table.columns = columns
        try:
            table = table.drop(index=0).reset_index(drop=True)
        except:
            table = table.reset_index(drop=True)

        try:
            table = table.drop(columns='index')
        except:
            pass

        return table, response

    def execute(self, python_text: str, question: str, content: str):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        error = False
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(python_text.strip())
            except Exception as e:
                error = True
                print("Generated python function is not executable. Falling back to cot...")
                print(e)

        if error:
            result = ask_openai([
                {
                    "role": "system",
                    "content": prompt_fallback_python.format(question=question, content=content),
                }
            ])
        else:
            result = stdout_buffer.getvalue()

        # in case error is True, there's no need to apply the last step "final answer:" after python execution, because it is already done during the error handling
        # so inside the caller function, do not launch the final LLM call if error is True
        return result, error

    def table_normalization(self, query: str, intermediate_tables: dict[str, list[pd.DataFrame]]) -> str:
        html_tables = []
        for key, tables in intermediate_tables.items():
            for table in tables:
                html_tables.append(table.to_html(index=False))

        prompt_text = "\n\n".join(html_tables)
        prompt = prompt_normalization.format(question=query, tables=prompt_text)
        list_of_rules_raw = ask_openai([
            {
                "role": "system",
                "content": prompt,
            }
        ])

        list_of_rules = self.remove_markdown_syntax(self.extract_result(list_of_rules_raw, "Final answer:"))
        return list_of_rules

    def table_insertion(self, texts: list[str], tables: Dict[Union[int, str], List[pd.DataFrame]]) -> List[str]:
        """
        Inserisce nelle stringhe 'texts' le tabelle HTML corrispondenti.
        'tables' può essere:
          - un dict con chiavi intere contigue (0..n-1) corrispondenti agli indici di texts, oppure
          - un dict con chiavi arbitrarie (es. nomi di settori o filenames). In questo caso si usa l'ordine di list(tables.values()).
        Se il numero di elementi in texts e in tables non coincide, la funzione fa un best-effort:
          - usa min(len(texts), len(tables_list)) e non tocca i testi in eccesso.
        """
        # Normalizza tables in una lista la cui i-esima entry corrisponde al texts[i]
        # Caso 1: chiavi numeriche contigue 0..n-1
        try:
            numeric_keys = all(isinstance(k, int) for k in tables.keys())
        except Exception:
            numeric_keys = False

        if numeric_keys:
            # Verifica che le chiavi siano contigue a partire da 0
            keys_sorted = sorted(tables.keys())
            if keys_sorted == list(range(len(keys_sorted))):
                tables_list = [tables[i] for i in range(len(keys_sorted))]
            else:
                # Non contigue: trasformiamo comunque in lista ordinata per chiave crescente
                tables_list = [tables[k] for k in keys_sorted]
        else:
            # Chiavi non numeriche: usa ordine di inserimento / values()
            tables_list = list(tables.values())

        new_texts = []
        n = min(len(texts), len(tables_list))
        # Per i testi oltre n, lasciamo il testo originale
        for i, text in enumerate(texts):
            new_text = text
            if i < n:
                tables_for_text = tables_list[i] or []
                for j in range(len(tables_for_text)):
                    # protezione: assicurati che la placeholder esista nel testo prima di sostituire
                    placeholder = f"<Table{j + 1}>"
                    if placeholder in new_text:
                        # converti la table in html (index=False)
                        try:
                            new_text = new_text.replace(placeholder, tables_for_text[j].to_html(index=False))
                        except Exception as e:
                            print(f"DEBUG: errore convertendo table[{i}][{j}] in html: {e}", flush=True)
                            new_text = new_text.replace(placeholder, "")  # fallback: rimuovi placeholder
                    else:
                        # placeholder non presente: potremmo comunque appendere la tabella o ignorare; per ora ignoriamo
                        pass
            else:
                print(f"DEBUG: Nessuna tabella disponibile per texts[{i}] - lascio il testo originale.", flush=True)

            new_texts.append(new_text)

        return new_texts

    def query(self, query: str, tables: Dict[int, List[pd.DataFrame]], texts: List[str]) -> str:
        """
        given a query and a list of tables, this function processes each table in this way:
        - Filtering: extraction of relevant rows and columns from each table
        - Table normalization: definition of the rule to change values across different units of measurements. This is done in a single LLM call with all the tables and the query.
        - Table insertion: the tables are re-inserted back into the page text
        - PoT: the LLM generates the Python code to answer the question
        - Python execution: execute the Python code
        - Final answer: the final result is given back to the LLM, which produces a general response explaining the answer
        """

        intermediate_responses = {}
        intermediate_tables = {}
        error = False

        # filter table
        for key, tables in tables.items():
            intermediate_tables[key] = []
            intermediate_responses[key] = []
            for table in tables:
                filtered_table, extract_response = self.filter_table(query, table)
                if isinstance(filtered_table, int) and filtered_table == -1:
                    error = True

                if error:
                    intermediate_responses[key].append(-1)
                    intermediate_tables[key].append(-1)
                else:
                    intermediate_responses[key].append(extract_response)
                    intermediate_tables[key].append(filtered_table)

        # normalize table
        list_of_rules = self.table_normalization(query, intermediate_tables)

        # Table insertion
        new_texts = self.table_insertion(texts, intermediate_tables)

        # PoT
        prompt = prompt_total.format(question=query, paragraph="\n\n".join(new_texts) + "\n\n" + list_of_rules)
        python_text_raw = ask_openai([
            {
                "role": "system",
                "content": prompt,
            }
        ])

        python_code = self.remove_markdown_syntax(self.extract_result(python_text_raw, "Final answer:"))

        results, error = self.execute(python_code, query, '\n\n'.join(new_texts) + "\n\n" + list_of_rules)

        if error:
            return python_text_raw + '\n' + 'Output :\n' + results

        # Final answer
        results = self.remove_markdown_syntax(self.extract_result(results, "Final answer:"))

        if results is not None:
            final_string = "🧠Program of Thought in action\n" + python_text_raw + "\n\nResult :\n" + results

        else:
            final_string = "🧠Program of Thought in action\n" + python_text_raw

        print("final_string:\n", final_string)

        return final_string

"""
if __name__ == '__main__':
    ag = QueryAgent()

    df1 = pd.read_csv('/home/n284480/GRI-QA/table_dataset/OTC_SU_2023/310_1.csv', sep=';')
    df2 = pd.read_csv('/home/n284480/GRI-QA/table_dataset/OTC_SU_2023/311_0.csv', sep=';')
    df3 = pd.read_csv('/home/n284480/GRI-QA/table_dataset/OTC_SU_2023/311_1.csv',  sep=';')

    tables = {
        0: [df1, df2, df3]
    }
    '''
    df1 = pd.DataFrame({
        "index": [0, 1, 2],
        "name": ["A", "B", "C"],
        "value": [5, 15, 25]
    })

    df2 = pd.DataFrame({
        "index": [0, 1, 2],
        "name": ["X", "Y", "Z"],
        "value": [7, 12, 30]
    })

    df3 = pd.DataFrame({
        "index": [0, 1, 2],
        "name": ["E", "F", "G"],
        "value": [5, 15, 25]
    })

    tables = {
        0: [df1, df2],
        1: [df3]
    }
    
    texts = [
        "Per rispondere alla domanda, considera i dati riportati nelle tabelle " +
        " e ".join([f"<Table{i + 1}>" for i in range(len(tables[0]))]) +
        " e analizza i valori principali.",

        "Infine, fai riferimento alla tabella " + "<Table1>" + " per completare l'analisi."
    ]
    '''

    texts = [
        "Per rispondere alla domanda, considera i dati riportati nelle tabelle " +
        " e ".join([f"<Table{i + 1}>" for i in range(len(tables[0]))]) +
        " e analizza i valori principali.",
    ]

    result = ag.query("What is the reduction in estimated total energy consumption from 2022 to 2023?", tables, texts)
    print(result)
"""
