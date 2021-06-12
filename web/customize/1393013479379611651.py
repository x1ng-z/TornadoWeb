def main(input_data, context):
    mv1 = input_data['mv1']['value']
    mv2 = input_data['mv2']['value']
    mv1_k = input_data['mv1_k']['value']

    if mv1_k >= 1 :
        mv1_k = 1
    mv_o = mv1 * mv1_k + mv2 * ( 1 - mv1_k )

    out = {
        "mv_o":{'value':mv_o}
    }
    return out
    