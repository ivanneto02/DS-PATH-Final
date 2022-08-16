def proceed():
    # take in user choice
    u_in = input("Proceed? (\"yes\" or \"no\"): ")

    # do not move on until the user types an answer
    while u_in != "yes" or u_in != "no":
        u_in = input("You need to type \"yes\" or \"no\".")

    # process the user choice
    if u_in == "yes":
        return True
    else:
        return False