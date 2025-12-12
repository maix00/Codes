import DataMinkBasics
from ContractRollover import ContractRollover
import pandas as pd

product_id_list = ['A']
data_mink_path = '../data/data_mink_product_2025/'
product_contract_start_end_path = '../data/wind_mapping.csv'
product_contract_start_end = pd.read_csv(product_contract_start_end_path)
product_contract_start_end.columns.values[0] = "PRODUCT"
product_contract_start_end.columns.values[1] = "CONTRACT"

for product_id in product_id_list:
    detector = DataMinkBasics.RolloverDetector()
    data = product_contract_start_end[product_contract_start_end['PRODUCT'].str.startswith(product_id + '.')]
    detector.add_data_table('product_contract_start_end', data)
    rollovers = detector.detect_rollover_points(path=data_mink_path)
    for rollover in rollovers:
        print(rollover, '\n')

    
# End of file futures_data_mink/futures_data_mink_check_7.py