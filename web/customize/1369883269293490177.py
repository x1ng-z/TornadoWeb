def main(input_data, context_data):
    #input_data['pv']['value']
    if 'pv' not in context_data:
        context_data['pv']=1
        context_data['mv']=1
        context_data['ff']=1
    context_data['pv']+=1
    #context_data['mv']+=1
    context_data['ff']+=1
    if context_data['pv']>100:
         context_data['pv']=0
         context_data['ff']=0
    out = {
        "pv":{'value':context_data['pv']},
        "mv":{'value':context_data['mv']},
        "ff":{'value':context_data['ff']}
    }
    return out