def map_comp(a, b):
    return (a > b) * 2 + (a == b) * 1 - 1

def get_state(obs):
    locs = [[0, 0]] * 4
    taxi_row, taxi_col, locs[0][0], locs[0][1], locs[1][0], locs[1][1], locs[2][0], locs[2][1], locs[3][0], locs[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    state = []

    for loc in locs:
        state.append(map_comp(taxi_row, loc[0]))
        state.append(map_comp(taxi_col, loc[1]))
    
    state.append(obstacle_north)
    state.append(obstacle_south)
    state.append(obstacle_east)
    state.append(obstacle_west)
    state.append(passenger_look)

    return state