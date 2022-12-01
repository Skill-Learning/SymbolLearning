# key = str of action_vector + obj_type

ACTION_DICT = {
    "100000" : "PokeX",
    "010000" : "PokeY",
    "001000" : "GraspTop",
    "000100" : "GraspFront",
    "000010" : "PokeTop",
}




LABELS_DICT = {
    "PokeX+std_cube" : 0,
    "PokeY+std_cube" : 1,
    "GraspTop+std_cube" : 2,
    "GraspFront+std_cube" : 3,
    "PokeTop+std_cube" : 4,
    "PokeX+high_cube" : 5,
    "PokeY+high_cube" : 6,
    "GraspTop+high_cube" : 7,
    "GraspFront+high_cube" : 8,
    "PokeTop+high_cube" : 9,
    "PokeX+long_cube" : 10,
    "PokeY+long_cube" : 11,
    "GraspTop+long_cube" : 12,
    "GraspFront+long_cube" : 13,
    "PokeTop+long_cube" : 14,
    "PokeX+wide_cube" : 15,
    "PokeY+wide_cube" : 16,
    "GraspTop+wide_cube" : 17,
    "GraspFront+wide_cube" : 18,
    "PokeTop+wide_cube" : 19    
}