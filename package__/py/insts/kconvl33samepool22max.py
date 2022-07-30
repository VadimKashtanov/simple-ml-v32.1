from math import exp, tanh

activate = [
	lambda x: 1 / (1 + exp(-x)),
	tanh,
	lambda x: exp(-x*x),
	lambda x: x * (x >= 0) 
]

localderiv = [
	lambda x: activate[0](x) * (1 - activate[0](x)),
	lambda x: 1 - tanh(x)**2,
	lambda x: -2*x*activate[2](x),
	lambda x: (x >= 0)
]

class KCONVL33SAMEPOOL22MAX(Inst):
	_id = 2
	ID = 2

	name='KCONVL33SAMEPOOL22MAX'
	params_names=['Ax', 'Ay', 'n0', 'n1', 'activ', 'input_start', 'ystart', 'wstart', 'locdstart', 'drop_rate']

	def check(self):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return Ax%2 == 0 and Ay%2==0 and n0 > 0 and n1 > 0 and activ in (0,1,2,3) and 100 >= drop_rate >= 0 and all(i>=0 and i==int(i) for i in params)
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		Yx, Yy = int(Ax/2), int(Ay/2)
	
		for _n1 in range(n1):
			for y in range(Yy):
				for x in range(Yx):
					_00, _10, _01, _11 = [0,0,0,0]
					for _n0 in range(n0):
						for _x in range(3):
							for _y in range(3):
								if (__y:=(y*2+_y-1))>=0 and (__x:=(x*2+_x-1))>=0:
									_00 += var[l*total + input_start + _n0*Ax*Ay + __y*Ax + __x] * w[wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y-1))>=0 and (__x:=(x*2+_x))<Ax:
									_10 += var[l*total + input_start + _n0*Ax*Ay + __y*Ax + __x] * w[wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y))<Ay and (__x:=(x*2+_x-1))>=0:
									_01 += var[l*total + input_start + _n0*Ax*Ay + __y*Ax + __x] * w[wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y))<Ay and (__x:=(x*2+_x)) < Ax:
									_11 += var[l*total + input_start + _n0*Ax*Ay + __y*Ax + __x] * w[wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
					
					_00 += w[wstart + n1*n0*9 + _n1*Ax*Ay + (y*2)*Ax + x*2]
					_10 += w[wstart + n1*n0*9 + _n1*Ax*Ay + (y*2)*Ax + x*2+1]
					_01 += w[wstart + n1*n0*9 + _n1*Ax*Ay + (y*2+1)*Ax + x*2]
					_11 += w[wstart + n1*n0*9 + _n1*Ax*Ay + (y*2+1)*Ax + x*2+1]
					
					_00, _10, _01, _11 = list(map(activate[activ], (_00, _10, _01, _11)))
					_max = max([_00,_10,_01,_11])
					
					'''if _max == _00: var[l*total + ystart + _n1*Yx*Yy + y*Yx + x] = _00
					elif _max == _10: var[l*total + ystart + _n1*Yx*Yy + y*Yx + x] = _10
					elif _max == _01: var[l*total + ystart + _n1*Yx*Yy + y*Yx + x] = _01
					else: var[l*total + ystart + _n1*Yx*Yy + y*Yx + x] = _11'''
					var[l*total + ystart + _n1*Yx*Yy + y*Yx + x] = _max
					#can do it with max and .index			
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float]):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		Yx, Yy = int(Ax/2), int(Ay/2)
		
		imgstart = line*sets*total + _set*total + input_start
	
		for _n1 in range(n1):
			for y in range(Yy):
				for x in range(Yx):
					_00, _10, _01, _11 = 0,0,0,0
					for _n0 in range(n0):
						for _y in range(3):
							for _x in range(3):
								if (__y:=(y*2+_y-1)) >=0 and (__x:=(x*2+_x-1)) >= 0:
									_00 += var[imgstart + _n0*Ax*Ay + __y*Ax + __x] * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y-1)) >= 0 and (__x:=(x*2+_x)) < Ax:
									_10 += var[imgstart + _n0*Ax*Ay + __y*Ax + __x] * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y))<Ay and (__x:=(x*2+_x-1)) >= 0:
									_01 += var[imgstart + _n0*Ax*Ay + __y*Ax + __x] * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
								if (__y:=(y*2+_y)) < Ay and (__x:=(x*2+_x)) < Ax:
									_11 += var[imgstart + _n0*Ax*Ay + __y*Ax + __x] * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
	
					_00 += w[ws*_set + wstart + n1*n0*9 + _n1*Ax*Ay + (y*2)*Ax + x*2]
					_10 += w[ws*_set + wstart + n1*n0*9 + _n1*Ax*Ay + (y*2)*Ax + x*2+1]
					_01 += w[ws*_set + wstart + n1*n0*9 + _n1*Ax*Ay + (y*2+1)*Ax + x*2]
					_11 += w[ws*_set + wstart + n1*n0*9 + _n1*Ax*Ay + (y*2+1)*Ax + x*2+1]
	
					l_00, l_10, l_01, l_11 = list(map(localderiv[activ], (_00, _10, _01, _11)))
					_00, _10, _01, _11 = list(map(activate[activ], (_00, _10, _01, _11)))
	
					_max = max([_00,_10,_01,_11])
					if _max == _00:
						var[line*sets*total + _set*total + ystart + _n1*Yx*Yy + y*Yx + x] = _00
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x] = l_00
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x + 1] = 0
					elif _max == _10:
						var[line*sets*total + _set*total + ystart + _n1*Yx*Yy + y*Yx + x] = _10
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x] = l_10
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x + 1] = 1
					elif _max == _01:
						var[line*sets*total + _set*total + ystart + _n1*Yx*Yy + y*Yx + x] = _01
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x] = l_01
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x + 1] = 2
					else:
						var[line*sets*total + _set*total + ystart + _n1*Yx*Yy + y*Yx + x] = _11
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x] = l_11
						locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x + 1] = 3
						#The locd has 2 floats for each Y cells => 2*Ycell
					#can do it with max and .index
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, line:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		Yx, Yy = int(Ax/2), int(Ay/2)
	
		imgstart = line*sets*total + _set*total + input_start
	
		for _n1 in range(n1):
			for y in range(Yy):
				for x in range(Yx):
					dlds = grad[line*sets*total + _set*total + ystart + _n1*Yx*Yy + y*Yx + x] * locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x]
					max_id = int(locd[line*sets*locds + _set*locds + locdstart + _n1*2*Yx*Yy + y*2*Yx + 2*x + 1])
					pool_x, pool_y = max_id % 2, int((max_id-(max_id%2))/2)  #pool_x (0'th or 1'st col), pool_y (0'th or 1'st row)
					
					meand[ws*_set + wstart + n1*n0*9 + _n1*Ax*Ay + (y*2+pool_y)*Ax + x*2+pool_x] += dlds
	
					for _n0 in range(n0):
						for _x in range(3):
							for _y in range(3):
								if max_id == 0:
									if (__y:=(y*2+_y-1))>=0 and (__x:=(x*2+_x-1))>=0:
										grad[imgstart + _n0*Ax*Ay + __y*Ax + __x] += dlds * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
										meand[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x] += dlds * var[imgstart + _n0*Ax*Ay + __y*Ax + __x]
								if max_id == 1:
									if (__y:=(y*2+_y-1))>=0 and (__x:=(x*2+_x))<Ax:
										grad[imgstart + _n0*Ax*Ay + __y*Ax + __x] += dlds * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
										meand[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x] += dlds * var[imgstart + _n0*Ax*Ay + __y*Ax + __x]
								if max_id == 2:
									if (__y:=(y*2+_y))<Ay and (__x:=(x*2+_x-1))>=0:
										grad[imgstart + _n0*Ax*Ay + __y*Ax + __x] += dlds * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
										meand[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x] += dlds * var[imgstart + _n0*Ax*Ay + __y*Ax + __x]
								if max_id == 3:
									if (__y:=(y*2+_y))<Ay and (__x:=(x*2+_x)) < Ax:
										grad[imgstart + _n0*Ax*Ay + __y*Ax + __x] += dlds * w[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x]
										meand[ws*_set + wstart + _n1*9*n0 + _n0*9 + _y*3 + _x] += dlds * var[imgstart + _n0*Ax*Ay + __y*Ax + __x]
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return int((Ax/2*Ay/2)*n1)
	
	def buildstackmodel_weights(self):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return n0*n1*9 + (Ax*Ay)*n1
	
	def buildstackmodel_locds(self):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return int(2*n1*(Ay/2)*(Ax/2))
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return [(f'{_id}.Y [kconvl33samepool22max]',stack_start)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return [(f'{_id}.K [kconvl33samepool22max]',stack_start), (f'{_id}.B [kconvl33samepool22max]',stack_start + n0*n1*9)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay, n0,n1, activ,input_start,ystart,wstart,locdstart,drop_rate = list(self.params.values())
		return [(f'{_id}.Y [kconvl33samepool22max]',stack_start)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_kconvl33samepool22max = "Ax","Ay", "n0","n1", "activ", "drate"
	
	requiredposition_kconvl33samepool22max = 1,1,1,1,1,0,0,0,0,1 #1,1,1 == Ax,Yx,activ
	
	def setupparamsstackmodel_kconvl33samepool22max(self, ystart, wstart, lstart, required):
		Ax,Ay, n0,n1, activ,drate = required
		return Ax,Ay, n0,n1, activ,istart,ystart,wstart,lstart,drate