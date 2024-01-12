import quaternionic
import spherical
import numpy as np
import quaternion


def get_rotation_matrix(q : [quaternionic.array, list])->np.ndarray:
	return q.to_rotation_matrix

def get_quaternion(rot_mat : np.ndarray)->quaternionic.array:
	return quaternionic.array.from_rotation_matrix(rot_mat)

def get_general_operations(
	q1 : np.quaternion,
	q2 : np.quaternion,
	dict : dict,
	type : str
	)->[np.quaternion,str]:
	try:
		perform_ops = f"q1{dict[type]}q2"
		qout = eval(perform_ops)
		return qout
	except Exception as e:
		return type

if __name__=="__main__":
	inp = [1,2,3,4]
	q1 = quaternionic.array(inp)
	rot_mat = get_rotation_matrix(q1)
	print(rot_mat)

	q2 = get_quaternion(rot_mat)
	print(q2)

	qa,qb = q1,q2
	dict = {
		"add":"+", 
		"subtract":"-", 
		"multiply":"*", 
		"divide":"/",
		"equal":"==", 
		"not_equal":"!=", 
		
	}
	rv = get_general_operations(q1,q1,dict,"equal")
	print(rv)