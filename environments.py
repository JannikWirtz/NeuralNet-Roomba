
# starts at bottom left corner
def addSquare(env, x,y,w,h):

    env.append([(x+w, y+h), (x+w, y)])
    env.append([(x+w, y+h), (x, y+h)])
    env.append([(x, y), (x+w, y)])
    env.append([(x, y), (x, y+h)])
    return env

def getEasyLevel():
    env = []
    # Largest Square (fence)
    addSquare(env, 0, 0, 100, 100)

    # single obstacle
    addSquare(env, 40, 40, 20, 20)

    return env

def getHardLevel():
    env = []
    
    # Largest Square (fence)
    addSquare(env, 0, 0, 100, 100)

    # couch corner
    addSquare(env, 80, 20, 10, 60)
    addSquare(env, 50, 20, 30, 10)
    
    # Table
    addSquare(env, 30, 50, 20, 20)

    return env


def getMediumLevel():
    env = []
    
    # Largest Square (fence)
    addSquare(env, 0, 0, 100, 100)

    addSquare(env, 80, 20, 10, 60)
    addSquare(env, 50, 20, 30, 10)
    
    addSquare(env, 10, 10, 10, 80)
    addSquare(env, 10, 10, 40, 10)
    addSquare(env, 20, 45, 40, 10)

    addSquare(env, 80, 80, -30, -10)
    return env

def getNarrowPathLevel():
    env = []
    
    # Largest Square (fence)
    addSquare(env, 0, 0, 100, 100)

    addSquare(env, 10, 0, 10, 80)
    addSquare(env, 30, 20, 10, 80)
    addSquare(env, 50, 0, 10, 80)
    addSquare(env, 70, 20, 10, 80)
    addSquare(env, 90, 0, 10, 80)

    return env

def getRoomLevel():
    env = []
    
    # Largest Square (fence)
    addSquare(env, 0, 0, 100, 100)

    addSquare(env, 0, 20, 40, 10)
    addSquare(env, 30, 60, 10, 20)
    addSquare(env, 70, 40, 10, 20)

    return env
    
def getLabyrinth():
    env = []
    env.append([(0, 50), (80, 50)])
    env.append([(0, 30), (25, 30)])
    env.append([(40, 50), (40, 15)])
    env.append([(10, 15), (40, 15)])
    env.append([(60, 0), (60, 30)])
    env.append([(10, 80), (20, 80)])
    env.append([(40, 80), (100, 80)])
    env.append([(80, 15), (100, 15)])
    env.append([(20, 70), (20, 100)])
    return env


if __name__ == '__main__': 
    import train
    train()