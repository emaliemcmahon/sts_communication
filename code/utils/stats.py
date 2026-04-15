
def p2star(p):
    if 0.001 > p: 
        star = '***'
    elif 0.01 > p >= 0.001:
        star = '**' 
    elif 0.05 > p >= 0.01:
        star = '*'
    else: 
        star = None
    return star
