import humanize
import datetime as dt
def humanization(n):
    return {
        "intcomma":humanize.intcomma(n),
        "intword":humanize.intword(n),
        "apnumber":humanize.apnumber(n),
    }

def humanizationDatetime(datetime):
    return {
        "naturalday":humanize.naturalday(datetime),
        "naturaldate":humanize.naturaldate(datetime),
        "naturaltime":humanize.naturaltime(datetime),
    }
def humanizationFileSize(size):
    return {
        "MB":humanize.naturalsize(size),
        "KiB":humanize.naturalsize(size,binary=True),
        "K":humanize.naturalsize(size,gnu=True),
    }

def humanizationFractional(num,den):
    return {
        "fractional":humanize.fractional(num/den)
    }

def humanizationScientific(x):
    return {
        "scientific":humanize.scientific(x)
    }
def humanizationDecimalConverter(fraction):
    return {
        "fractional":humanize.fractional(fraction),
        "scientific":humanize.scientific(fraction)
    }
def humanizeOtherToEnglish(n,datetime,size,fraction,language="ru_RU"):
    _t = humanize.i18n.activate(language)
    x1,x2,x3 = humanize.intcomma(n),humanize.intword(n),humanize.apnumber(n)
    x11,x22,x33 = humanize.naturalday(datetime),humanize.naturaldate(datetime),humanize.naturaltime(dt.timedelta(seconds=3))
    MB,KiB,K=humanize.naturalsize(size),humanize.naturalsize(size,binary=True),humanize.naturalsize(size,gnu=True),
    f,s  = humanize.fractional(fraction),humanize.scientific(fraction)
    r1 = {"f":f,"s":s,"x1":x1,"x2":x2,"x3":x3,"x11":x11,"x22":x22,"x33":x33,"MB":MB,"KiB":KiB,"K":K}
    humanize.i18n.deactivate()
    x1,x2,x3 = humanize.intcomma(n),humanize.intword(n),humanize.apnumber(n)
    x11,x22,x33 = humanize.naturalday(datetime),humanize.naturaldate(datetime),humanize.naturaltime(dt.timedelta(seconds=3))
    MB,KiB,K=humanize.naturalsize(size),humanize.naturalsize(size,binary=True),humanize.naturalsize(size,gnu=True),
    f,s  = humanize.fractional(fraction),humanize.scientific(fraction)
    r2 = {"f":f,"s":s,"x1":x1,"x2":x2,"x3":x3,"x11":x11,"x22":x22,"x33":x33,"MB":MB,"KiB":KiB,"K":K}
    
    return r1,r2



    
    
if __name__=="__main__":
    n = 10
    datetime = dt.date(2007, 6, 5)
    size = 100
    fraction = 0.33
    rv1,rv2 = humanizeOtherToEnglish(n,datetime,size,fraction,language="ru_RU")
    print(f"{rv1} to {rv2}")

