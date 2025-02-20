# All assertion functions print "Success!" or a failure message.

def printAndAssertEQ(val, target):
    try:
        assert val == target
        print("Success!")
    except:
        print(f"Equals assertion failed!\nTarget: {target}\nValue: {val}")

def printAndAssertRange(val, lower_bound, upper_bound):
    try:
        assert lower_bound <= val
        print("Success!")
    except:
        print(f"Lower Bound Failed!\nLower Bound: {lower_bound}\nValue: {val}")
    try:
        assert val <= upper_bound
        print("Success!")
    except:
        print(f"Upper Bound Failed!\nUpper Bound: {upper_bound}\nValue: {val}")