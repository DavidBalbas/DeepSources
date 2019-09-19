import os
	
try:
    os.makedirs('7mindata/npymats')
    os.makedirs('7mindata/real_planck')
    os.makedirs('7mindata/fullmaps')
    os.makedirs('5mindata/npymats')
    os.makedirs('5mindata/fullmaps')
    os.makedirs('5minplanedata/npymats')
    os.makedirs('5minplanedata/fullmaps')
	    
    print("Creation succesful")
except FileExistsError:
    print("Directory/ies already exists") 
