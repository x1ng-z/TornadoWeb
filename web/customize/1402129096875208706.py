def main(input_data, context_data):
    apc_value_1 = input_data['apc_value_1']['value']
    apc_value_2 = input_data['apc_value_2']['value']
    out = {
        "mv":{'value':apc_value_1}
    }
    context_data['result_his'] += (apc_value_2 + apc_value_1+2)
    return out