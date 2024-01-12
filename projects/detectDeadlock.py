
if __name__=="__main__":
    import deadlock_detector_new
    import threading

    deadlock_detector_new.init()

    
    lists = [
        ("l0",threading.Lock()),
        ("l1",threading.Lock()),
        ("l2",threading.Lock()),
        ("l3",threading.Lock())
    ]
    deadlock_detection_for_name = "l3"
    def random_walk_with_deadlockInfo(lists,deadlock_detection_for_name):
        s = set(lists)
        rv = ""
        i= 0 
        for name,st in s:
            print(name)
            rv+=":: initial random walk :: "
            
            rv += f"->{name} "
            i+=1
            print(rv)

            if st.acquire()==True:
                if name == deadlock_detection_for_name:
                    print(st.acquire())
                print(f"acquired {name}")
            else:
                print(f"not acquired {name}")
            print(f"release {st.release()}")
        rv= "final random walk :- " + rv
        return rv
    walk = random_walk_with_deadlockInfo(lists,deadlock_detection_for_name)
    print(walk)