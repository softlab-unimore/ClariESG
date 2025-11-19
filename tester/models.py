import pandas as pd
from prompts.query_agent_prompts import prompt_extract, prompt_fallback_python, prompt_normalization, prompt_total, prompt_total_gpt5
from typing import Union, List, Dict
import io
import re
import contextlib
from openai import OpenAI
from dotenv import load_dotenv
import traceback

load_dotenv()

class QueryAgent:
    def __init__(self):
        self.client = OpenAI()
        self.model_name = "gpt-5-mini" #"gpt-4o-mini"

    def ask_openai(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0 if "5" not in self.model_name else 1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Errore durante la chiamata a OpenAI: {str(e)}"

    def remove_markdown_syntax(self, text: str) -> str:
        # Remove triple backtick code blocks (```python ... ```)
        text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", '', m.group()), text)

        # Remove inline code (`code`)
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Remove bold (**text** or __text__)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Remove italic (*text* or _text_)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        # Remove blockquotes
        text = re.sub(r"^>\s?", '', text, flags=re.MULTILINE)

        text = text.replace("python", "")
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
        response = self.ask_openai([
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
        columns = [el for i,el in enumerate(table.iloc[0,:]) if i in rows_columns_extracted["columns"]]

        table = table[table["0"].isin(rows_columns_extracted["rows"])] # table["0"] is "index"
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
                traceback.print_exc(file=stderr_buffer)

        if error:
            print(stderr_buffer.getvalue(), flush=True)
            result = self.ask_openai([
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

    def table_normalization(self, query: str, tables: list[pd.DataFrame]) -> str:
        html_tables = []
        for table in tables:
            html_tables.append(table.to_html(index=False))

        prompt_text = "\n\n".join(html_tables)
        prompt = prompt_normalization.format(question=query, tables=prompt_text)
        list_of_rules_raw = self.ask_openai([
            {
                "role": "system",
                "content": "You are an helpful assistant that extracts normalization rules from tables.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ])

        list_of_rules = self.remove_markdown_syntax(self.extract_result(list_of_rules_raw, "Final answer:"))
        return list_of_rules

    def table_insertion(self, texts: list[str], tables: Dict[int, List[pd.DataFrame]]) -> List[str]:
        new_texts = []
        for i, text in enumerate(texts):
            new_text = text
            for j in range(len(tables[i])):
                new_text = new_text.replace(f"<Table{j + 1}>", tables[i][j].to_html(index=False))
            new_texts.append(new_text)

        return new_texts

    def query(self, query: str, tables: List[pd.DataFrame], company_names: List[str]) -> str:
        """
        given a query and a list of tables, this function processes each table in this way:
        - Filtering: extraction of relevant rows and columns from each table
        - Table normalization: definition of the rule to change values across different units of measurements. This is done in a single LLM call with all the tables and the query.
        - Table insertion: the tables are re-inserted back into the page text
        - PoT: the LLM generates the Python code to answer the question
        - Python execution: execute the Python code
        - Final answer: the final result is given back to the LLM, which produces a general response explaining the answer
        """

        intermediate_tables = []
        error = False

        for table in tables:
            intermediate_tables.append(table)
            """filtered_table, extract_response = self.filter_table(query, table)
            if isinstance(filtered_table, int) and filtered_table == -1:
                error = True

            if error:
                intermediate_tables.append(table)
            else:
                intermediate_tables.append(filtered_table)"""

        # normalize table
        list_of_rules = self.table_normalization(query, intermediate_tables)

        tables_txt = [f"Company name: {company_name}\n{table.to_html()}" for table, company_name in zip(intermediate_tables, company_names)]
        input_to_py_programmer = "\n\n".join(tables_txt) + "\n\n" + list_of_rules

        # PoT
        if "5" in self.model_name:
            prompt = prompt_total_gpt5.format(question=query, paragraph=input_to_py_programmer)
        else:
            prompt = prompt_total.format(question=query, paragraph=input_to_py_programmer)

        python_text_raw = self.ask_openai([
            {
                "role": "system",
                "content": "You are an expert data analyst and Python programmer specialized in data extraction from HTML tables.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ])

        python_code = self.remove_markdown_syntax(self.extract_result(python_text_raw, "Final answer:"))
        print(python_code)
        results, error = self.execute(python_code, query, input_to_py_programmer)

        if error:
            print("Error in pot parsing, falling back to cot...")
            results = self.remove_markdown_syntax(self.extract_result(results, "Final answer:"))
            return results

        # final answer
        #results = self.remove_markdown_syntax(self.extract_result(results, "Final answer:"))
        return results