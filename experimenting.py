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
def _(con, mo):
    _df = mo.sql(
        f"""
        show tables
        """,
        engine=con
    )
    return


@app.cell
def _(con, iot_23, mo):
    _df = mo.sql(
        f"""
        describe iot_23
        """,
        engine=con
    )
    return


if __name__ == "__main__":
    app.run()
