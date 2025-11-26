import gradio as gr
import os
import json
import subprocess
from itertools import islice
from urllib.parse import quote
import sys
import llm
import pandas as pd
import shutil
import gradio_actions
import markdown2
from gradio_toggle import Toggle
import re
from query_agent import QueryAgent
from pathlib import Path

server_host = "155.185.48.176"  # se lavoro sul server unimore, sennò server_host = 'localhost'

messages = [
    {"role": "system",
     "content": (
         "You are an experienced assistant in sustainability and GRI standards. "
         "You help users understand the data extracted from the PDFs of various companies."
         "Instructions: "
         "- Think step by step through the information before answering (use reasoning internally). "
         "- Answer clearly, concisely and succinctly. "
         "- Use only the data provided in the context (text extracted from the page + data tables for each company being analysed)."
         "- If you cannot find the answer from the context, say so clearly. "
         "- Indicate the row, cell and name of the company you used for the answer (from the table within the text) and indicate the PAGE and NUMBER of the table (from the context) also for every extracted information"
         "- Do not explain your reasoning process; just give the final answer and required details."
     )},
    {"role": "user",
     "content": (
         "Here are the name of the file (comapny name) and the relevant context like text of the page with tables:\n---\n{context}\n---\n"
         "Now, answer the following question based strictly on the context.\n\nQuestion: {user_message}"
     )}
]
messages_sectors = [
    {
        "role": "system",
        "content": (
            "You are an expert assistant specialized in sustainability reporting and GRI standards. "
            "You help users analyze and compare sustainability data across multiple companies belonging to the same sector.\n\n"
            "Instructions:\n"
            "- Think carefully through the provided information before answering.\n"
            "- Answer clearly, concisely, and without unnecessary explanation.\n"
            "- Use **only** the information contained in the provided context "
            "(which includes the extracted text and tables for each company within the sector).\n"
            "- If the answer cannot be found in the context, say so explicitly.\n"
            "- For every factual element you mention, indicate:\n"
            "   • The SECTOR name\n"
            "   • The COMPANY name (source)\n"
            "   • The PAGE number and TABLE number (as provided in the context)\n"
            "   • The specific row or cell in the table, if applicable\n"
            "- Do **not** explain your reasoning process; only provide the final answer with precise references."
        )
    },
    {
        "role": "user",
        "content": (
            "Below you have the extracted context from several companies belonging to the same sector. "
            "Each section specifies the sector, company name (source), and the related text and tables.\n\n"
            "---\n{context}\n---\n\n"
            "Now, based strictly on this information, answer the following question:\n\n"
            "Question: {user_message}"
        )
    }
]


def clear_all():
    return None


def load_companies_with_summary(companies_name, base_path=Path("./table_dataset")):
    companies_data = {}

    for name in companies_name:
        summary_path = os.path.join(base_path, name, "summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read().strip()
        else:
            summary_text = ""
        companies_data[name] = summary_text

    return companies_data


def filter_companies_by_sector(selected_sectors):
    """
    selected_sectors: lista di settori scelti nella checkbox.
                      Se vuota → mostra tutte le aziende.
    """
    # Mostra tutto se nessun settore selezionato
    if not selected_sectors:
        companies = gradio_actions.get_docs_from_db()
        companies_dict = load_companies_with_summary(companies)
        return render_cards_from_dict(companies_dict)

    filtered_companies = gradio_actions.filter_company_cards(selected_sectors)
    # Carico le summary solo delle aziende filtrate
    companies_dict = load_companies_with_summary(filtered_companies)
    return render_cards_from_dict(companies_dict)


def upload_and_process_files(files):
    """
    Function that receives a list of PDF files, executes calls to main.py, and returns text on the GRI values found.
    It now also extracts the company name and updates the PDF names.
    """
    csv_group.visible = False
    if not files:
        return "⚠️No file uploaded"

    # Percorso del file di query
    json_file_query = os.path.join('json_config', 'en_queries_301-308.json')
    with open(json_file_query, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    for file in files:

        orig_filepath = file.name  # original local path
        orig_filename = os.path.basename(orig_filepath)

        pdf_path_tmp = os.path.join("reports", orig_filename)
        shutil.copy(orig_filepath, pdf_path_tmp)  # server copy

        company_name = llm.get_company_name(pdf_path_tmp)
        company_name_clean = re.sub(r'[^\w_-]', '_', company_name.strip()) if company_name else "UNKNOWN"

        # new file name
        pdf_basename_orig = os.path.splitext(orig_filename)[0]
        pdf_basename = f"{pdf_basename_orig}-{company_name_clean}"
        pdf_name_server = os.path.abspath(os.path.join("reports", f"{pdf_basename}.pdf"))
        os.rename(pdf_path_tmp, pdf_name_server)

        try:
            # 1. Dense
            subprocess.run(
                [sys.executable, "main.py", "--pdf", pdf_name_server, "--embed", "--use_dense"],
                shell=False,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )

            # 2. Sparse
            subprocess.run(
                [sys.executable, "main.py", "--pdf", pdf_name_server, "--embed", "--use_sparse"],
                shell=False,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )

            # 3. Ensemble with query
            subprocess.run(
                [sys.executable, "main.py", "--pdf", pdf_name_server, "--load_query_from_file", json_file_query, "--use_ensemble"],
                shell=False,
                check=True,
                env=env,
                capture_output=True,
                text=True
            )

        except subprocess.CalledProcessError as e:
            results.append(
                f"📁**{pdf_basename_orig}** -- company name **{company_name}**: Error while executing main.py  \n"
                f"stdout:\n{e.stdout}\n\nstderr:\n{e.stderr}"
            )
            continue

        new_metadata_path = os.path.join("table_dataset", pdf_basename, "metadata_after_llm.json")

        with open(new_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        output_lines = [f"📁**{pdf_basename_orig}** -- company name **{company_name}** \n "]

        for gri_code, description in data.items():
            if gri_code in metadata:
                gri_line = f"   🔹**GRI {gri_code}**: {description}  "
                output_lines.append(gri_line)
                seen = set()
                for page, other_num in metadata[gri_code]:
                    key = (page, other_num)
                    if key not in seen:
                        seen.add(key)
                        page += 1
                        link = f"   [pag.{page}](http://{server_host}:8080/viewer.html?file={quote(pdf_basename)}.pdf#page={page})  "
                        output_lines.append(f"     {link} -> {page - 1}_{other_num}.csv  ")

        results.append("\n".join(output_lines))

    return "\n\n".join(results)


def handle_chat_with_pdf(chat_history, chat_input_data, docs_list, sectors_list, select_pot_value):
    """
    Handles a user query with selected PDF files (docs_list).
    If `select_pot_value` is active, it uses QueryAgent (Program-of-Thought); otherwise, it follows the classic flow with direct LLM.
    If the user wants to query a sector instead, sectors_list will be an input to the function.
    """
    user_message = chat_input_data.get("text", "").strip()

    if len(docs_list) == 0 and len(sectors_list) == 0:
        response = "⚠️ No documents selected or no sectors selected."
        return chat_history + [{"role": "assistant", "content": response}]

    if user_message == "":
        response = "⚠️ No input received from User."
        return chat_history + [{"role": "assistant", "content": response}]

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    sectors = False
    if len(sectors_list) > 0:
        sectors = True

    # === CASE 1: Program of Thought ===
    if select_pot_value:

        # === CASE 1A: PoT + sectors ===
        if sectors:
            ag = QueryAgent()
            all_sector_tables = {}

            for sector in sectors_list:
                print(f"Sector processing: {sector}")
                try:
                    subprocess.run(
                        [sys.executable, "main.py", "--query", user_message, "--use_ensemble", "--sectors", sector],
                        shell=False, check=True, env=env, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    return [{"role": "assistant",
                             "content": f"⚠️ Error during PoT for sector '{sector}':\n{e.stderr}"}]

                # Percorso dei metadata del settore
                metadata_path = os.path.join("sectors", sector, "verbal_questions_metadata.json")
                if not os.path.exists(metadata_path):
                    return [{"role": "assistant", "content": f"⚠️ No metadata found for sector '{sector}'"}]

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if user_message not in metadata:
                    return [{"role": "assistant",
                             "content": f"⚠️ No reference found for query “{user_message}” in sector '{sector}'"}]

                refs = metadata[user_message]  # list of dict with source and pages
                sector_sources = {}  # here begins the new dictionary for sources

                for source_entry in refs:
                    src_name = source_entry.get("source")
                    source_tables = [] # list of tables for this source

                    for page_entry in source_entry.get("pages", []):
                        csv_files = page_entry.get("csv_files", [])
                        for csv_filename in csv_files:
                            csv_path = os.path.join("table_dataset", src_name, csv_filename)
                            if os.path.exists(csv_path):
                                try:
                                    df = pd.read_csv(csv_path, sep=";")
                                    source_tables.append(df)
                                except Exception:
                                    print(f"DEBUG: CSV reading error {csv_path}")

                    if source_tables:
                        sector_sources[src_name] = source_tables  # add tables for the source
                if sector_sources:
                    all_sector_tables[sector] = sector_sources  # add to sector

            if not all_sector_tables:
                response = "⚠️ No tables found in the selected sectors."
                return chat_history + [{"role": "assistant", "content": response}]

            # Build texts for PoT
            texts = []
            for sector, sources in all_sector_tables.items():
                source_names = list(sources.keys())
                text = (
                        f"Sector: {sector}\n"
                        f"The following reports were analysed: {', '.join(source_names)}.\n"
                        "Consider the data shown in the following tables: "
                        + " e ".join([f"<Table{i + 1}>" for i in range(sum(len(t) for t in sources.values()))])
                        + ". Analyse the main values to answer the question."
                )
                texts.append(text)

            try:
                result, intermediate_filtered_idx = ag.query(user_message, all_sector_tables, texts)
            except Exception as e:
                result = f"⚠️ Error during QueryAgent in sectors: {e}"

            return chat_history + [{"role": "assistant", "content": result}]

        # === CASE 1B: PoT ===
        else:
            ag = QueryAgent()
            all_tables = {}

            for file_idx, file in enumerate(docs_list):

                pdf_name = os.path.join(os.path.abspath(os.getcwd()), "reports", file + ".pdf")

                try:
                    subprocess.run(
                        [sys.executable, "main.py", "--pdf", pdf_name, "--query", user_message, "--use_ensemble"],
                        shell=False, check=True, env=env, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    return [{"role": "assistant", "content": f"⚠️ Error during PDF processing:\n{e.stderr}"}]

                metadata_path = os.path.join("table_dataset", file, "verbal_questions_metadata.json")
                if not os.path.exists(metadata_path):
                    return [{"role": "assistant", "content": f"⚠️ No metadata.json found for {file}"}]

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                file_tables = []
                refs = metadata[user_message]

                for page, num in refs:

                    # Search for CSV files in the folder
                    csv_file = os.path.join("table_dataset", file, f"{page}_{num}.csv")
                    if os.path.exists(csv_file):
                        try:
                            df = pd.read_csv(csv_file, sep=";")
                            file_tables.append(df)
                        except Exception:
                            print(f"DEBUG: Error during CSV reading {csv_file}")

                if len(file_tables) > 0:
                    all_tables[file] = file_tables

            if not all_tables:
                response = "⚠️ No tables found in the selected PDFs."
                return chat_history + [{"role": "assistant", "content": response}]

            texts = []
            for file, tables in all_tables.items():
                text = (
                        f"File: {file}\n"
                        "Consider the data shown in the following tables"
                        + " e ".join([f"<Table{i + 1}>" for i in range(len(tables))])
                        + ". Analyse the main values to answer the question."
                )
                texts.append(text)

            try:
                result, intermediate_filtered_idx = ag.query(user_message, all_tables, texts)
            except Exception as e:
                result = f"⚠️ Error during QueryAgent execution: {e}"

            return chat_history + [{"role": "assistant", "content": result}]

    # === CASE 2: CoT ===
    else:
        context = ""
        # === CASE 2A: CoT + sectors ===
        if sectors:
            try:
                subprocess.run(
                    [sys.executable, "main.py", "--query", user_message, "--use_ensemble", "--sectors"] + sectors_list,
                    shell=False, check=True, env=env, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                return [{"role": "assistant", "content": f"⚠️ Error during processing:\n{e.stderr}"}]

            # Itera sui sectors
            for sector in sectors_list:
                sector_dir = os.path.join("sectors", sector)
                metadata_path = os.path.join(sector_dir, "verbal_questions_metadata.json")
                if not os.path.exists(metadata_path):
                    context += f"⚠️No metadata found for the sector '{sector}'\n"
                    continue

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if user_message not in metadata:
                    context += f"⚠️  No reference found for query “{user_message}” in sector '{sector}'\n"
                    continue

                refs = metadata[user_message]  # list of dicts: [{"source": ..., "pages": [...]}, ...]
                combined_text = ""
                seen_pages = set()

                for source_entry in refs:
                    src_name = source_entry.get("source")
                    for page_entry in source_entry.get("pages", []):
                        page = page_entry.get("page_n")
                        csv_files = page_entry.get("csv_files", [])

                        if page in seen_pages:
                            continue

                        txt_path = os.path.join("table_dataset", src_name, f"{page}.txt")
                        if not os.path.exists(txt_path):
                            print(f"DEBUG: Missing TXT file {txt_path}")
                            continue

                        with open(txt_path, "r", encoding="utf-8") as f:
                            txt_content = f.read()

                        # Replace CSV placeholder
                        placeholders = re.findall(r'\[TABLEPLACEHOLDER\s*(\d+)]', txt_content)
                        for placeholder_num in placeholders:
                            csv_file = os.path.join("table_dataset", src_name, f"{page}_{placeholder_num}.csv")
                            if os.path.exists(csv_file):
                                try:
                                    df = pd.read_csv(csv_file, sep=";")
                                    table_str = f"\n[Table from {src_name} - page {page}, num {placeholder_num}]\n" + df.to_csv(
                                        index=False)
                                except Exception:
                                    print(f"DEBUG: error while reading CSV {csv_file}")
                                    table_str = ""
                            else:
                                table_str = ""

                            txt_content = re.sub(rf'\[TABLEPLACEHOLDER\s*{placeholder_num}]', table_str, txt_content)

                        combined_text += f"\n---\n# Source: {src_name}\n# Page {page}\n{txt_content}\n"
                        seen_pages.add(page)

                header = f"Sector: {sector}\n"
                context += f"{header}\n{combined_text}\n---\n"

        # === CASE 2B: CoT ===
        else:

            for file in docs_list:
                pdf_name = os.path.join(os.path.abspath(os.getcwd()), "reports", file + ".pdf")

                try:
                    subprocess.run(
                        [sys.executable, "main.py", "--pdf", pdf_name, "--query", user_message, "--use_ensemble"],
                        shell=False, check=True, env=env, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    return [{"role": "assistant", "content": f"⚠️ Error during PDF processing:\n{e.stderr}"}]

                metadata_path = os.path.join("table_dataset", file, "verbal_questions_metadata.json")
                if not os.path.exists(metadata_path):
                    return [{"role": "assistant", "content": f"⚠️ No metadata.json found for {file}"}]

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                if user_message not in metadata:
                    context += f"⚠️ No references found for query '{user_message}' in {file}\n"
                    continue

                refs = metadata[user_message]
                combined_text = ""
                pages = []

                for page, num in refs:

                    if page not in pages:
                        txt_path = os.path.join("table_dataset", file, f"{page}.txt")
                        if not os.path.exists(txt_path):
                            print(f"DEBUG: Missing TXT file {txt_path}")
                            continue

                        with open(txt_path, "r", encoding="utf-8") as f:
                            txt_content = f.read()

                        placeholders = re.findall(r'\[TABLEPLACEHOLDER\s*(\d+)]', txt_content)
                        for placeholder_num in placeholders:
                            csv_file = os.path.join("table_dataset", file, f"{page}_{placeholder_num}.csv")
                            if os.path.exists(csv_file):
                                try:
                                    df = pd.read_csv(csv_file, sep=";")
                                    table_str = f"\n[Table from page {page}, num {placeholder_num}]\n" + df.to_csv(
                                        index=False)
                                except Exception:
                                    print(f"DEBUG: error while reading CSV {csv_file}")
                                    table_str = ""
                            else:
                                table_str = ""
                            txt_content = re.sub(rf'\[TABLEPLACEHOLDER\s*{placeholder_num}]', table_str, txt_content)

                        combined_text += f"\n---\n# Page {page}\n{txt_content}\n"
                        pages.append(page)

                header = f"Company name: {file}\n"
                context += f"{header}\n{combined_text}\n---\n"

        if sectors:
            message = [
                messages_sectors[0],  # system
                {
                    "role": "user",
                    "content": messages[1]["content"].format(
                        context=context,
                        user_message=user_message
                    ),
                },
            ]
        else:
            message = [
                messages[0],  # system
                {
                    "role": "user",
                    "content": messages[1]["content"].format(
                        context=context,
                        user_message=user_message
                    ),
                },
            ]
        response = llm.ask_openai(message)
        return chat_history + [{"role": "assistant", "content": response}]


def make_card_html(company_name, summary_text):

    return f"""
    <div class="card">
        <div class="card-header">{company_name}</div>
        <div class="card-content">
            {markdown2.markdown(summary_text)}
        </div>
    </div>
    """


def render_cards_from_dict(companies_dict):
    cards_html_content = "".join(make_card_html(name, summary) for name, summary in companies_dict.items())
    return f"""
    <div class="cards-container">
        {cards_html_content}
    </div>
    """


def render_cards():
    companies = gradio_actions.get_docs_from_db()
    companies_dict = load_companies_with_summary(companies)
    return render_cards_from_dict(companies_dict)


with gr.Blocks() as chatbot_ui:
    gr.Markdown(
        "<h2 style='text-align: center; font-size: 40px;'>ClariESG</h2>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                min_height=500,
                max_height=500,
                avatar_images=(None, "./images/icon_chatbot.png"),
                show_copy_button=True,
                show_copy_all_button=True,
                group_consecutive_messages=False
            )

            chat_input = gr.MultimodalTextbox(
                elem_id='chat_input',
                interactive=True,
                placeholder="Enter question for the selected file...",
                show_label=False,
                sources="microphone",
            )

        with gr.Column(scale=1):
            with gr.Row():
                docs_list = gr.CheckboxGroup(
                    elem_id="docs_list",
                    choices=[],
                    label="Select documents to query",
                    value=[],
                    interactive=True,

                )
            with gr.Row():
                sectors_list = gr.CheckboxGroup(
                    elem_id="sectors_list",
                    choices=[],
                    label="Select the sectors to query ",
                    value=[],
                    interactive=True,

                )

            # Mutual exclusion logic between docs_list and sectors_list

            def handle_docs_change(selected_docs):
                # If there are selected documents, disable sectors_list
                if selected_docs:
                    return gr.update(interactive=False)
                else:
                    return gr.update(interactive=True)


            def handle_sectors_change(selected_sectors):
                # If there are selected sectors, disable docs_list
                if selected_sectors:
                    return gr.update(interactive=False)
                else:
                    return gr.update(interactive=True)

            docs_list.change(handle_docs_change, inputs=docs_list, outputs=sectors_list)
            sectors_list.change(handle_sectors_change, inputs=sectors_list, outputs=docs_list)

            with gr.Row(elem_id="row_toggle"):
                select_pot = Toggle(
                    elem_id='select_pot',
                    label='PoT',
                    show_label=False,
                    info='PoT',
                    value=False,
                    interactive=True,
                    color='#50B596',
                    transition=1
                )


    def clear_textbox():
        return {"text": ""}


    def disable_docs():
        return gr.update(interactive=False)


    def enable_docs():
        return gr.update(interactive=True)


    def disable_sectors():
        return gr.update(interactive=False)


    def enable_sectors():
        return gr.update(interactive=True)


    def disable_textbox():
        return gr.update(interactive=False)


    def disable_toggle():
        return gr.update(interactive=False)


    def enable_textbox():
        return gr.update(interactive=True)


    def enable_toggle():
        return gr.update(interactive=True)


    chat_msg = chat_input.submit(
        llm.add_user_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input, chat_input]
    )

    chat_msg.then(
        disable_docs,
        outputs=[docs_list]
    )

    chat_msg.then(
        disable_sectors,
        outputs=[sectors_list]
    )

    chat_msg.then(
        disable_textbox,
        outputs=[chat_input]
    )

    chat_msg.then(
        disable_toggle,
        outputs=[select_pot]
    )

    bot_msg = chat_msg.then(
        handle_chat_with_pdf,
        inputs=[chatbot, chat_input, docs_list, sectors_list, select_pot],
        outputs=[chatbot]
    )

    bot_msg.then(
        enable_docs,
        outputs=[docs_list]
    )

    bot_msg.then(
        enable_sectors,
        outputs=[sectors_list]
    )

    bot_msg.then(
        enable_textbox,
        outputs=[chat_input]
    )

    bot_msg.then(
        enable_toggle,
        outputs=[select_pot]
    )

    chat_msg.then(
        clear_textbox,
        outputs=[chat_input]
    )

    chatbot.like(gradio_actions.print_like_dislike, None, None, like_user_message=False)

with gr.Blocks() as company_cards:
    sectors = gradio_actions.get_sectors_from_db()

    with gr.Row(elem_id="cards-layout"):
        with gr.Column(scale=1):
            sector_selector_cards = gr.CheckboxGroup(
                elem_id='sector_selector_cards',
                choices=sectors,
                label="Filter by sector",
                interactive=True
            )

        with gr.Column(scale=3):
            cards_container = gr.HTML(elem_id="cards-zone")

    cards_container.value = render_cards()

    sector_selector_cards.change(
        fn=filter_companies_by_sector,
        inputs=sector_selector_cards,
        outputs=cards_container
    )

with gr.Blocks() as process_file_ui:
    gr.Markdown(
        "<h2 style='text-align: center; font-size: 40px;'>Extraction of GRI Information</h2>"
    )
    with gr.Row():

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="Load PDFs",
                file_types=[".pdf"],
                file_count="multiple",
                elem_id='pdf_input'
            )

            with gr.Row():
                clear_button = gr.Button(value="Clear")
                upload_button = gr.Button(value="Submit", variant='primary')

            output_box = gr.Markdown(
                label="Output",
                height=300,
                show_label=True,
                container=True,
                elem_id='output_box'
            )

        with gr.Column(scale=2):
            with gr.Group(visible=True) as csv_group:
                pdf_dropdown = gr.Dropdown(choices=[], value=None, label="🗂️ Select folder")
                csv_dropdown = gr.Dropdown(choices=[], value=None, label="📄 Select CSV file")

                dataframe = gr.Dataframe(visible=False, interactive=True, value=pd.DataFrame(), max_height=280,
                                         wrap=True, show_copy_button=True, show_search='search', label="CSV content")
                log_output = gr.Textbox(label="Output", interactive=False, autoscroll=False)

            with gr.Row():
                gr.HTML("")
                save_button = gr.Button(value="Save changes", variant='primary')
                gr.HTML("")

            pdf_dropdown.change(gradio_actions.list_csv_files, inputs=pdf_dropdown, outputs=csv_dropdown)
            csv_dropdown.change(gradio_actions.load_csv, inputs=[pdf_dropdown, csv_dropdown], outputs=dataframe)
            save_button.click(gradio_actions.save_csv, inputs=[pdf_dropdown, csv_dropdown, dataframe],
                              outputs=log_output)

    upload_button.click(
        fn=upload_and_process_files,
        inputs=pdf_input,
        outputs=output_box
    ).then(
        fn=gradio_actions.refresh_pdf_folders,
        inputs=[],
        outputs=pdf_dropdown
    ).then(
        fn=gradio_actions.update_docs_list,
        inputs=[],
        outputs=docs_list
    ).then(
        fn=gradio_actions.update_sectors_list,
        inputs=[],
        outputs=sectors_list
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=pdf_input
    )

if __name__ == "__main__":
    # Set the folder to serve
    pdf_dir = os.path.join(os.getcwd(), "reports")

    # Start an HTTP server in the background on the reports folder on port 8080.
    subprocess.Popen(
        ["python", "-m", "http.server", "8080"],
        cwd=pdf_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    with gr.Blocks(
            theme='lone17/kotaemon',
            title="ClariESG",
            css_paths="style.css",

    ) as demo:
        gr.TabbedInterface(
            [chatbot_ui, process_file_ui, company_cards],
            ["Chatbot", "Process File", "Company Card"],
        )

        demo.load(concurrency_limit=None, fn=render_cards, inputs=[], outputs=[cards_container])

        demo.load(concurrency_limit=None, fn=gradio_actions.refresh_pdf_folders, inputs=[], outputs=[pdf_dropdown])

        demo.load(concurrency_limit=None, fn=gradio_actions.refresh_docs_list, inputs=[], outputs=[docs_list])

        demo.load(concurrency_limit=None, fn=gradio_actions.refresh_sectors_list, inputs=[], outputs=[sectors_list])

    demo.launch()
