        
def get_movment(slope_error,y_error,rotation_coe = 1,movement_coe = 1):
    if slope_error is not None:
        mxv = 40
        if slope_error > mxv:
            slope_error = mxv
        elif slope_error <-mxv:
            slope_error = -mxv
        err = slope_error/mxv
        movement_vector = (10*5*(1-abs(err)),-(1-abs(err))*y_error)
        rotation_speed = slope_error
    else:
        movement_vector = (4*20,0)
        rotation_speed = 0
    return rotation_speed*rotation_coe, movement_vector*movement_coe