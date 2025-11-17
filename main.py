from utils import init_args
from runnable import Runnable
from table_extraction import UnstructuredTableExtractor
from tqdm import tqdm
import shutil
import os
import csv
import json
from itertools import islice
from bs4 import BeautifulSoup
import re
import llm
import build_summary_company

if __name__ == "__main__":

    args = init_args()
    r = Runnable(args)

    if len(args["load_query_from_file"]) > 0:

        with open(args["load_query_from_file"], 'r') as file:
            data = json.load(file)

        if os.path.isdir(args["pdf"]):
            file_names = os.listdir(args["pdf"])
        elif os.path.isfile(args["pdf"]):
            file_names = [args["pdf"]]
        else:
            raise ValueError("wrong file name")

        for file_name in file_names:
            if not file_name.lower().endswith(".pdf"):
                continue

            file_path = args["pdf"]
            base_name = os.path.basename(file_path)
            dir_name = os.path.splitext(base_name)[0]

            args["pdf"] = file_name
            metadata_path = os.path.join("table_dataset", dir_name)
            os.makedirs(metadata_path, exist_ok=True)

            gri_code_to_page = {}
            tables_as_html = set()

            for gri_code, description in islice(data.items(), 2, 17):  # from 3 to 17 (GRI)
                if gri_code not in gri_code_to_page:
                    gri_code_to_page[gri_code] = []

                args["query"] = description

                r.set_args(args)
                s = r.run()

                ute = UnstructuredTableExtractor("yolox", "hi_res")

                for doc in tqdm(s[:args["k"]]):  # top-k pages
                    tables, text_without_tables = ute.extract_table_unstructured([doc])

                    page_num = doc.metadata["page"]

                    # Save CSV tables
                    for i, table in enumerate(tables):
                        html_table = table[0].metadata.text_as_html
                        gri_code_to_page[gri_code].append((page_num, i))

                        # Convert HTML table to rows
                        soup = BeautifulSoup(html_table, "html.parser")
                        rows = []
                        for tr in soup.find_all("tr"):
                            cells = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
                            rows.append(cells)

                        csv_path = os.path.join(metadata_path, f"{page_num}_{i}.csv")
                        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerows(rows)

                    for elem in text_without_tables:
                        page = elem[2]
                        page_text = elem[0]
                        txt_path = os.path.join(metadata_path, f"{page}.txt")
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(page_text)

            with open(os.path.join(metadata_path, "metadata.json"), 'w', encoding='utf-8') as json_file:
                json.dump(gri_code_to_page, json_file, indent=4, ensure_ascii=False)

            pdf_basename = dir_name
            # I use openAI to skim through the tables found. For each GRI-table_taken_from_csv, I ask if it is relevant.
            new_metadata_path = llm.check(folder_path=os.path.join(".", "table_dataset"),  gri_code_list_path=args["load_query_from_file"], pdf_basename=pdf_basename)

            if not os.path.exists(new_metadata_path):
                results.append(f"📁{pdf_basename}: {new_metadata_path} not founded  ")
                continue

            x = os.path.join(".", "table_dataset", pdf_basename, "metadata.json")
            y = os.path.join(".", "table_dataset", pdf_basename, "metadata_before_llm.json")
            os.replace(x, y)

            new_name = os.path.join(".", "table_dataset", pdf_basename, "metadata_after_llm.json")
            os.replace(new_metadata_path, new_name)
            # I use openAI to format tables
            llm.formatted(folder_path=os.path.join(".", "table_dataset"), pdf_basename=pdf_basename)

            # I delete .txt files that do not have any associated tables
            csvs = [f for f in os.listdir(metadata_path) if f.endswith(".csv")]

            removed_txt = 0
            removed_csv = 0

            for filename in os.listdir(metadata_path):
                if not filename.endswith(".txt"):
                    continue

                page_name = os.path.splitext(filename)[0]  # es. "12" from "12.txt"
                txt_path = os.path.join(metadata_path, filename)

                #  search for csv files beginning with ‘page_name_’ (e.g. ‘12_0.csv’)
                has_csv = any(re.match(rf"^{page_name}_[0-15]+\.csv$", csv) for csv in csvs)

                if not has_csv:
                    os.remove(txt_path)
                    removed_txt += 1
                    print(f"Deleted TXT without table: {txt_path}")

            build_summary_company.build_summary(dir_name)

    elif len(args["query"]) > 0:

        # search by sector
        if args["sectors"] is not None and len(args["sectors"]) > 0:
            s = r.run(args["sectors"])
            question_to_page = {}
            question_to_page[args["query"]] = []

            for sector in args["sectors"]:
                sector_dir = os.path.join("sectors", sector)
                os.makedirs(sector_dir, exist_ok=True)

                # group by source
                source_to_pages = {}

                for doc in tqdm(s[:args["k"]]):
                    source_path = doc.metadata.get("source", "")
                    page_n = doc.metadata.get("page", None)
                    if not source_path or page_n is None:
                        continue

                    source_name = os.path.splitext(os.path.basename(source_path))[0]
                    source_dir = os.path.join("table_dataset", source_name)
                    dest_dir = os.path.join(sector_dir, source_name)
                    os.makedirs(dest_dir, exist_ok=True)

                    if not os.path.exists(source_dir):
                        continue

                    txt_src = os.path.join(source_dir, f"{page_n}.txt")
                    if not os.path.exists(txt_src):
                        continue

                    csv_files = [f for f in os.listdir(source_dir) if re.match(fr"{page_n}_\d+\.csv$", f)]
                    if not csv_files:
                        continue

                    # copy TXT and CSV
                    shutil.copy(txt_src, os.path.join(dest_dir, f"{page_n}.txt"))
                    for csv_file in csv_files:
                        shutil.copy(os.path.join(source_dir, csv_file), os.path.join(dest_dir, csv_file))

                    # add to the structure source->pages
                    if source_name not in source_to_pages:
                        source_to_pages[source_name] = []

                    source_to_pages[source_name].append({
                        "page_n": page_n,
                        "csv_files": csv_files
                    })

                # update question_to_page
                for source_name, pages in source_to_pages.items():
                    question_to_page[args["query"]].append({
                        "source": source_name,
                        "pages": pages
                    })

                metadata_path = os.path.join(sector_dir, "verbal_questions_metadata.json")
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            existing_data = {}
                else:
                    existing_data = {}

                if args["query"] in existing_data:
                    existing_data[args["query"]].extend(question_to_page[args["query"]])
                else:
                    existing_data.update(question_to_page)

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=4)

            print("✅ TXT and CSV copied and metadata updated with source and pages.")

        else:

            if os.path.isdir(args["pdf"]):
                file_names = os.listdir(args["pdf"])
            elif os.path.isfile(args["pdf"]):
                file_names = [args["pdf"]]
            else:
                raise ValueError(f"wrong file name")

            for file_name in file_names:
                splitted_file_name = file_name.split(".")
                if splitted_file_name[-1] != "pdf":
                    continue

                file_path = args["pdf"]
                base_name = os.path.basename(file_path)
                dir_name = os.path.splitext(base_name)[0]
                args["pdf"] = file_name

                question_to_page = {}
                tables_as_html = set()

                question_to_page[args["query"]] = []
                csvs = [f for f in os.listdir(os.path.join("table_dataset", dir_name)) if f.endswith(".csv")]

                matched_csvs = []

                s = r.run()

                for doc in tqdm(s[:args["k"]]):  # keeps only the top k pages with the highest score, where k is specified in the Python command (default = 5)
                    page_str = str(doc.metadata["page"])
                    for csv_file in csvs:
                        p = int(re.search(r"(\d+)_\d+\.csv$", csv_file).group(1))
                        if page_str == str(p):
                            matched_csvs.append(os.path.join(os.path.join("table_dataset", dir_name), csv_file))
                            i = int(re.search(r"_(\d+)\.csv$", csv_file).group(1))
                            question_to_page[args["query"]].append((p, i))

                metadata_path = os.path.join("table_dataset", dir_name, "verbal_questions_metadata.json")

                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

                if not os.path.exists(metadata_path):
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        f.write("{}")
                        existing_data = {}

                else:
                    with open(metadata_path, 'r') as json_file:
                        try:
                            existing_data = json.load(json_file)
                        except json.JSONDecodeError:
                            existing_data = {}

                existing_data.update(question_to_page)

                with open(metadata_path, "w", encoding='utf-8') as json_file:
                    json.dump(existing_data, json_file, indent=4)

    else:
        s = r.run()
