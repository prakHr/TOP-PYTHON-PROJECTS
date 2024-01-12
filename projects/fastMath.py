import pymath

def pow_set(listt):
    try:
        if len(set(listt))<=21:
            return pymath.powerset(set(listt))
        else:
            return "elements are > 21"
    except Exception as e:
        return str(e)

def COMBINATION(n,k):
    import sys
    sys.set_int_max_str_digits(100000)
    try:
        return pymath.C(n,k)
    except Exception as e:
        return str(e)

def PERMUTATION(n,k):
    import sys
    sys.set_int_max_str_digits(100000)
    try:
        return pymath.P(n,k)
    except Exception as e:
        return str(e)

def LCM(a,b):
    try:
        return pymath.xlcm(a,b)
    except Exception as e:
        return str(e)

def GCD(a,b):
    try:
        return pymath.xgcd(a,b)
    except Exception as e:
        return str(e)

if __name__=="__main__":
    listt = list(['a','B','A'])
    ans = pow_set(listt)
    print(ans)

    n,k = 200000,100000
    ans = COMBINATION(n,k)
    print(ans)

    n,k = 200,100
    ans = PERMUTATION(n,k)
    print(ans)

    
    
    a = 6
    d = {
        123:"YES",
        22:"NO",
        33:"YES"
    }
    ans = 1
    b = []
    for k,v in d.items():
        if v=="YES":
            b.append(k)
    if len(b)!=0:
        ans = GCD(a,b[0])
        lans = LCM(a,b[0])
        for ele in b:
            ans = GCD(a,ele)
            lans = LCM(a,ele)
        print(ans)
        print(lans)


    n = 10000
    fns = [
        "pymath.prime_factors(n)",
        "pymath.largest_prime_factor(n)",
        "pymath.coprimes(n)",
        "pymath.phi(n)",
    ]
    for fn_name in fns:
        print(fn_name)
        print(eval(fn_name))

    n = 10
    rest_fns = [
        'is_composite', 
        'is_even', 
        'is_highly_composite', 
        'is_largely_composite',
        'is_mersenne_prime', 
        'is_odd', 
        'is_perfect', 
        'is_prime', 
        'is_regular',
        'primes',  
        'primes_up_to',
        
    ]
    for rest_fn in rest_fns:
        rest_fn = rest_fn+"(n)"
        print(rest_fn)
        print(eval(f"pymath.{rest_fn}"))

    num = "123"
    baseSrc = 34
    baseDst = 12
    if 2<=baseSrc<=36 and 2<=baseDst<=36:
        ans = pymath.convert_base(num,baseSrc,baseDst)
        print(ans)