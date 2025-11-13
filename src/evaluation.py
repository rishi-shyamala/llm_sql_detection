import duckdb
import pandas as pd

def evaluate(conn, query, column, value, target_score, total_rows):
    if column.lower() in query.lower() and value.lower() in query.lower():
        return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-5, 'f2':-5} 
    elif value.lower() in query.lower():
        return {'output': None, 'tp':0, 'fp':0, 'tn':0, 'fn':0, 'precision':0, 'recall':0, 'f1':-2, 'f2':-2} 
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

    
if __name__ == '__main__':
    duckdb_db_path = "../data/data.duckdb"
    csv_path = '../evaluation/evaluation_test.csv'
    
    df = pd.read_csv(csv_path)
    con = duckdb.connect(duckdb_db_path)

    
    con.close()