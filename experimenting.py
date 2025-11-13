import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import duckdb
    con = duckdb.connect("data.duckdb")
    return (con,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(con):
    def evaluate(conn, query, column, value, target_score, total_rows):
        # if column.lower() in query.lower() and value.lower() in query.lower():
            # return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-5, 'f2':-5} 
        # elif value.lower() in query.lower():
            # return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-2, 'f2':-2} 
        count_query = f'with target_query as ({query.rstrip(';')}) select {column}, count(*) as count from target_query group by {column}'
        try:
            output = con.sql(count_query).fetchall()
            tp, fp, tn, fn = 0,0,0,0
            # print(count_query)
            output_map = {key:value for key, value in output}
            for category, score in output:
                if category == value:
                    tp = score
                else:
                    fp += score
            fn = target_score - tp
            tn = total_rows - tp - fp - fn
            precision = tp / (tp + fp) if tp+fp > 0 else 0.0
            recall = tp / (tp + fn) if tp+fn > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision+recall > 0) else 0.0
            beta = 1.5
            perfect = 1.3
            f2 = ((1 + (beta**2)) * precision * recall) / ((beta*precision) + recall) if (precision+recall > 0) else 0.0
            final_score=f2/perfect
            if precision > 0.8 and recall > 0.8:
                final_score += 1
                if precision > 0.95 and recall > 0.95:
                    final_score += 2
            return {'output': output_map, 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn, 'precision':precision, 'recall':recall, 'f1':f1, 'f2':final_score} 
        except Exception as e:
            return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-20, 'f2':-20} 
    return (evaluate,)


@app.cell
def _(mo):
    num_ssh = mo.ui.slider(0, 5897)
    num_ssh
    #TP
    return (num_ssh,)


@app.cell
def _(mo):
    num_portscan = mo.ui.slider(0, 243)
    num_portscan
    #FP1
    return (num_portscan,)


@app.cell
def _(mo):
    num_benign = mo.ui.slider(0, 10801)
    num_benign
    #FP2
    return (num_benign,)


@app.cell
def _(con, evaluate):
    nq = evaluate(con, 'select * from ids2017 where "Destination Port" = 22', 'Label', 'SSH-Patator', 5897, 3117165)
    print(nq)
    return


@app.cell
def _(con, evaluate):
    nq2 = evaluate(con, "select *, 'Protocol' as label from ids2017", 'Label', 'SSH-Patator', 5897, 3117165)
    print(nq2)
    return


@app.cell
def _(con, evaluate, num_benign, num_portscan, num_ssh):

    test_query = f"""(select * from ids2017 where Label = 'PortScan' limit {num_portscan.value} ) UNION (select * from ids2017 where Label = 'SSH-Patator' limit {num_ssh.value}) UNION (select * from ids2017 where Label = 'BENIGN' limit {num_benign.value})"""
    print(test_query)
    new_query = evaluate(con, test_query, 'Label', 'SSH-Patator', 5897, 3117165)
    print(new_query)
    print('score:', new_query['f2'])
    # con.sql('with query as (select * from ids2017 where "Destination Port" = 22) select label, count(*) as count from query group by label').fetchall()
    return


@app.cell
def _(con, evaluate):
    chatgpt_response1 = """SELECT *
    FROM ids2017
    WHERE "Destination Port" = 22
    AND "Flow Packets/s" >= (
    SELECT quantile_cont("Flow Packets/s", 0.99)
    FROM ids2017
    WHERE "Destination Port" = 22
    );"""
    print(evaluate(con, chatgpt_response1, 'Label', 'SSH-Patator', 5897, 3117165))
    return (chatgpt_response1,)


@app.cell
def _(chatgpt_response1):
    a = chatgpt_response1.rstrip(';')
    print(a)
    return


@app.cell
def _(con, ids2017, mo):
    _df = mo.sql(
        f"""
        select count(*) from ids2017
        """,
        engine=con
    )
    return


@app.cell
def _(con, ids2017, mo):
    _df = mo.sql(
        f"""
        with query as (select * from ids2017 where "Destination Port" = 22) select label, count(*) as count from query group by label
        """,
        engine=con
    )
    return


@app.cell
def _(con, ids2017, mo):
    _df = mo.sql(
        f"""
        select label, count(*) as count from ids2017 group by label
        """,
        engine=con
    )
    return


@app.cell
def _():
    import pandas as pd

    csv_path = 'evaluation/evaluation_test.csv'
    df = pd.read_csv(csv_path)
    #new_query = evaluate(con, test_query, 'Label', 'SSH-Patator', 5897, 3117165)
    return (df,)


@app.cell
def _(df):
    for q, w in df.iterrows():
        print(w)
    return


@app.cell
def _(con, df, evaluate):
    for r, c in df.iterrows():
        current_query = c['Output']
        current_value = c['Attack Type']
        current_target_value = c['Expected Rows']
        current_total_rows = c['Total Rows']
        current_label = c['Column']
        print('---------QUERY', r, '---------')
        print(evaluate(con, current_query, current_label, current_value, current_target_value, current_total_rows))
    return


@app.cell
def _():
    from datasets import load_dataset
    ds = load_dataset("motherduckdb/duckdb-text2sql-25k", split="train")
    ds[0]
    return (ds,)


@app.cell
def _():
    keep_categories = [
            "guides/sql_features/asof_join",
            "guides/sql_features/full_text_search",
            "sql/aggregates",
            "sql/case_sensitivity",
            "sql/configuration",
            "sql/constraints",
            "sql/data_types/bitstring",
            "sql/data_types/blob",
            "sql/data_types/boolean",
            "sql/data_types/date",
            "sql/data_types/enum",
            "sql/data_types/interval",
            "sql/data_types/list",
            "sql/data_types/map",
            "sql/data_types/nulls",
            "sql/data_types/numeric",
            "sql/data_types/overview",
            "sql/data_types/struct",
            "sql/data_types/text",
            "sql/data_types/time",
            "sql/data_types/timestamp",
            "sql/data_types/timezones",
            "sql/data_types/union",
            "sql/duckdb_table_functions",
            "sql/expressions/case",
            "sql/expressions/cast",
            "sql/expressions/collations",
            "sql/expressions/comparison_operators",
            "sql/expressions/in",
            "sql/expressions/logical_operators",
            "sql/expressions/overview",
            "sql/expressions/star",
            "sql/expressions/subqueries",
            "sql/functions/bitstring",
            "sql/functions/blob",
            "sql/functions/char",
            "sql/functions/date",
            "sql/functions/dateformat",
            "sql/functions/datepart",
            "sql/functions/enum",
            "sql/functions/interval",
            "sql/functions/nested",
            "sql/functions/numeric",
            "sql/functions/overview",
            "sql/functions/patternmatching",
            "sql/functions/time",
            "sql/functions/timestamp",
            "sql/functions/timestamptz",
            "sql/functions/utility",
            "sql/indexes",
            "sql/information_schema",
            "sql/introduction",
            "sql/query_syntax/filter",
            "sql/query_syntax/from",
            "sql/query_syntax/groupby",
            "sql/query_syntax/grouping_sets",
            "sql/query_syntax/having",
            "sql/query_syntax/limit",
            "sql/query_syntax/orderby",
            "sql/query_syntax/qualify",
            "sql/query_syntax/sample",
            "sql/query_syntax/select",
            "sql/query_syntax/setops",
            "sql/query_syntax/unnest",
            "sql/query_syntax/values",
            "sql/query_syntax/where",
            "sql/query_syntax/window",
            "sql/query_syntax/with",
            "sql/samples",
            "sql/statements/create_macro",
            "sql/statements/create_schema",
            "sql/statements/create_sequence",
            "sql/statements/pivot",
            "sql/statements/select",
            "sql/statements/unpivot",
            "sql/window_functions"
        ]
    return (keep_categories,)


@app.cell
def _(ds, keep_categories):
    df = ds.to_pandas()
    keep = df.category.isin(keep_categories)
    df[keep][['prompt', 'query', 'schema']]
    return (df,)


@app.cell
def _():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
            max_seq_length=4096,
            load_in_4bit=True,
            fast_inference=True,
            use_flash_attention_2=True,
            max_lora_rank=8,
            full_finetuning=False,
            gpu_memory_utilization=0.75,
            conservativeness=0.5,
        )
    return (tokenizer,)


@app.cell
def _(tokenizer):
    SYSTEM_PROMPT = (
            "You are an expert DuckDB SQL assistant.\n"
            "Take the user request, problem background, and SQL context and return a DuckDB-compatible SQL query.\n" 
            "Use any DuckDB clauses, functions, aggregations, or joins needed to make the best query possible\n"
        )
    chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '" + SYSTEM_PROMPT.replace("'", "\\'") + "' + eos_token }}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] + eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '' }}{% endif %}"
        )
    tokenizer.chat_template = chat_template
    return (SYSTEM_PROMPT,)


@app.cell
def _(tokenizer):
    print(tokenizer.chat_template)
    return


@app.cell
def _(SYSTEM_PROMPT, tokenizer):
    demo_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "You have a DuckDB table named logs with columns "
                    "(timestamp, src_ip, dst_ip, dst_port, bytes). "
                    "Write a query that returns all rows where dst_port = 22 "
                    "and bytes > 0."
                ),
            },
    ] 
    tmp = tokenizer.apply_chat_template(demo_messages, apply_generation_prompt=True, tokenize=False)

    return (tmp,)


@app.cell
def _(tmp):
    tmp
    return


if __name__ == "__main__":
    app.run()
