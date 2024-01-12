import flag


env = flag.string("env", "production", "The environment to use")

def CASE():
    return {
        "uppercase":env.upper(),
        "lowercase":env.lower(),
        "titlecase":env.title(),
        "reversetitlecase":env.lower()[0]+env.upper()[1:]
    }

if __name__ == '__main__':
    flag.parse()
    ans = CASE()
    print(ans)



