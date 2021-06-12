# import start
import numpy as np


### customer code start
def main(input_data, context):
    IN1 = input_data["IN1"]

    mv = IN1['mv']['value']
    pv = IN1['pv']['value']
    sp = IN1['sp']['value']
    mvup = IN1['mvup']['value']
    mvdown = IN1['mvdown']['value']
    ff = IN1['ff']['value'] if 'ff' in IN1 else 0
    kf = IN1['kf']['value'] if 'kf' in IN1 else 0
    kp = IN1['kp']['value']
    ki = IN1['ki']['value']
    kd = IN1['kd']['value']
    deadZone = IN1['deadZone']['value']
    dmvHigh = IN1['dmvHigh']['value']
    dmvLow = IN1['dmvLow']['value']
    # print(IN1)
    if 'sp' not in context:
        context['sp'] = sp
    if 'ff' not in context:
        context['ff'] = ff
    err = sp - pv
    if context['sp'] != sp:
        context['sp'] = sp
        context['residualList'] = [err]
    if 'residualList' not in context:
        context['residualList'] = [err]
    else:
        context['residualList'].append(err)
    if len(context['residualList']) >= 3:
        context['residualList'] = context['residualList'][-3:]
    partkp = 0
    partki = 0
    partkd = 0
    if len(context['residualList']) == 3:
        partkp = kp * (context['residualList'][-1] - context['residualList'][-2])
        partki = ki * context['residualList'][-1]
        partkd = kd * (context['residualList'][-1] - 2 * context['residualList'][-2] + context['residualList'][-3])
        delta_u = partkp + partki + partkd
    else:
        delta_u = 0

    delta_ff = ff - context['ff']
    # update ff
    context['ff'] = ff
    delta_u += delta_ff * kf
    # dmv limit
    if dmvHigh < delta_u:
        delta_u = dmvHigh
    elif -dmvHigh > delta_u:
        delta_u = -dmvHigh

    # mv deadzone
    if dmvLow > abs(delta_u):
        delta_u = 0

    # pv deadzone
    if ((sp - abs(deadZone)) < pv) and ((sp + abs(deadZone)) > pv):
        delta_u = 0
    update_mv = delta_u + mv
    # in limit
    if update_mv >= mvup:
        update_mv = mvup
    elif update_mv <= mvdown:
        update_mv = mvdown

    outpin = input_data['OUT1']
    outpinName = ''
    for key in outpin.keys():
        outpinName = outpin[key]['pinName']

    OUT1 = {
        'mv': {
            'pinname': outpinName,
            'value': update_mv,
            "partkp": partkp,
            "partki": partki,
            "partkd": partkd
        }
    }
    return OUT1
