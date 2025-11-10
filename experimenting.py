import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _(duckdb):

    con = duckdb.connect("data.duckdb")
    return (con,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(con):
    def evaluate(conn, query, column, value, target_score, total_rows):
        if column.lower() in query.lower() and value.lower() in query.lower():
            return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-5, 'f2':-5} 
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
            return {'output': output_map, 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn, 'precision':precision, 'recall':recall, 'f1':f1, 'f2':f2/(perfect)} 
        except Exception as e:
            return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-20, 'f2':-20, 'error':e} 
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
    nq = evaluate(con, "", 'Label', 'SSH-Patator', 5897, 3117165)
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

    new_query = evaluate(con, test_query, 'Label', 'SSH-Patator', 5897, 3117165)
    print(new_query)
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
def _(con, ids2017, mo):
    _df = mo.sql(
        f"""
        SELECT label, count(*) FROM ids2017 group by label
        """,
        engine=con
    )
    return


@app.cell
def _():
    import unsloth
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Instruct-2507",
        max_seq_length = 2048, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )
    return (tokenizer,)


@app.cell
def _(tokenizer):
    from unsloth.chat_templates import get_chat_template
    tokenizer2 = get_chat_template(
        tokenizer,
        chat_template = "qwen3-instruct",
    )
    return (tokenizer2,)


@app.cell
def _(tokenizer2):
    tokenizer2.get_chat_template()
    return


if __name__ == "__main__":
    app.run()
