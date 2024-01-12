def find_domain_name(lookup_codename):
    from world.database import Database
    db = Database()
    return db.lookup_code(lookup_codename)

def check(n):
    if n==None:return "Not found!"
    return n
def find_domain_names(lookup_codenames):
    rvs = [find_domain_name(lookup_codename) for lookup_codename in lookup_codenames]
    rvs = map(check,rvs)
    return list(rvs)


if __name__=="__main__":
    lookup_codenames = ["it","us","hi"]
    print(find_domain_names(lookup_codenames))