from math import exp, tanh

'''			lstm2d : [Ax,Ay] -> [Bx,Ay]
x:[Ax,Ay]

f0 = f(x@W + h[-1]@U + b)
f1 = f(x@W + h[-1]@U + b)
f2 = f(x@W + h[-1]@U + b)
g0 = g(x@W + h[-1]@U + b)
e = f0 * e[-1] + f1 * g0
h = f2 * e

h:[Bx:Ay]

f(x) = 1 / (1 + exp(-x))
g(x) = tanh(x)

Inputs = Ax*Ay
Outputs = (Bx*Ay)*2 (store e and h)

W:[Bx,Ax]
U:[Bx,Bx]
B:[Bx,Ay]
'''

'''	Var struct
e:[Bx,Ay]
h:[Bx,Ay]
'''

'''	Weight struct
Wf0:[Bx,Ax]
Uf0:[Bx,Bx]
Bf0:[Bx,Ay]

Wf1:[Bx,Ax]
Uf1:[Bx,Bx]
Bf1:[Bx,Ay]

Wf2:[Bx,Ax]
Uf2:[Bx,Bx]
Bf2:[Bx,Ay]

Wg0:[Bx,Ax]
Ug0:[Bx,Bx]
Bg0:[Bx,Ay]
'''

'''	Locd struct
f0:[Bx,Ay]
f1:[Bx,Ay]
f2:[Bx,Ay]
g0:[Bx,Ay]
'''

#	Params : [Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]

class LSTM2D(Inst):
	_id = 5
	ID = 5

	name='LSTM2D'
	params_names=['Ax', 'Ay', 'Bx', 'istart', 'ystart', 'wstart', 'locdstart', 'drate']

	def check(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return all(i >= 0 for i in params) and Ax > 0 and Ay > 0 and Bx > 0 and drate >= 0 and drate < 100
	
	def mdl(self,
		total:int, time:int,
		var:[float], w:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		inp = total*time + istart
		W = wstart
		out = total*time + ystart
	
		_W = Bx*Ay
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		Yx = Bx
		Yy = Ay
	
		for y in range(Yy):
			for x in range(Yx):
				#### f0,f1,f2 = f(x@W + h[-1]@W + b)
				f0,f1,f2,g0 = 0
				####
				# f0 is f0<x,y>
				####
	
				#### .W
				for k in range(X):
					vpos = inp + y*Ax + k
					wpos = (k*Bx + y)	##### Pas finis la position dans le Weight ### k*Bx + y
					f0 += var[vpos]*w[W + 0*lineW + wpos]
					f1 += var[vpos]*w[W + 1*lineW + wpos]
					f2 += var[vpos]*w[W + 2*lineW + wpos]
					g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .U
				if time > 0:
					for k in range(Y):
						#								   (e)		(h[y][x])
						vpos = total*(time-1) + ystart + (Bx*Ay) + (y*Ax + k) 
						wpos = _W + y*Bx + k
						f0 += var[vpos]*w[W + 0*lineW + wpos]
						f1 += var[vpos]*w[W + 1*lineW + wpos]
						f2 += var[vpos]*w[W + 2*lineW + wpos]
						g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .B
				wpos = _W + _U + (y*Bx + x)
				f0 += w[W + 0*wpos]
				f1 += w[W + 1*lineW + wpos]
				f2 += w[W + 2*lineW + wpos]
				g0 += w[W + 3*lineW + wpos]
	
				#### activ(_sum)
				f0 = logistic(f0)
				f1 = logistic(f1)
				f2 = logistic(f2)
				g0 = tanh(g0)
	
				#### e = f0 * e[-1] + f1 * g0
				#### l - 1 >= 0
				if time == 0:
					e = f1*g0
				else:
					e_1 = var[out - total + inp + i]
					e = f0*e_1 + f1*g0
					
				h = f2 * e
	
				var[out + (y*Bx + x)] = e
				var[out + Bx*Ay + (y*Bx + x)] = h
	
	'''
	=========== Forward ==================
	f0 = f(sf0 = xW + h[-1]U + b)
	f1 = f(sf1 = xW + h[-1]U + b)
	f2 = f(sf2 = xW + h[-1]U + b)
	g0 = f(sg0 = xW + h[-1]U + b)
	e = f0*e[-1] + f1*g0
	h = f2 * e
	
	========== Backward propagation (chain derivation) ============
	
	#h = f2 * e
		grad(f2) = grad(h) * e
		grad(e) += grad(h) * f2
	#e = f0*e[-1] + f1*g0
		grad(f0) = grad(e) * e[-1]
		grad(e[-1]) += grad(e) * f0
		grad(f1) = grad(e) * g0
		grad(g0) = grad(e) * f1
	#f0 = f(sf0 = xW + h[-1]U + b)
		grad(sf0) = grad(f0) * f'(sf0) = grad(f0) * f0 * (1 - f0)		#f = logistic; f' = logistic' = f*(1 - f)
	#f1 = f(sf1 = xW + h[-1]U + b)
		grad(sf1) = grad(f1) * f'(sf1) = grad(f1) * f1 * (1 - f1)
	#f2 = f(sf2 = xW + h[-1]U + b)
		grad(sf2) = grad(f2) * f'(sf2) = grad(f2) * f2 * (1 - f2)
	#g0 = f(sg0 = xW + h[-1]U + b)
		grad(sg0) = grad(g0) * f'(sg0) = grad(g0) * (1 - g0*g0)		#g = tanh; g' = tanh' = 1 - g^2
	
	=============== A shorter version ===============
	(the grad(e) have to be computed firste, because it's value will be use several times)
	
	grad(e) += grad(h) * f2
	grad(e[-1]) += grad(e) * f0
	grad(sf0) = grad(e) * e[-1] * f0 * (1 - f0) 
	grad(sf1) = grad(e) * g0 * f1 * (1 - f1)
	grad(sf2) = grad(h) * e * f2 * (1 - f2)
	grad(sg0) = grad(e) * f1 * (1 - g0*g0)
	
	================ C version ======================
	
	dh = grad[h]
	de = dh * f2 + grad[e]		#[time+1] had updated the gradient (we too for the line [time-1])
	
	grad[e] = de
	
	grad[e[-1] += de * f0
	dsf0 = de * e[-1] * f0 * (1 - f0)
	dsf1 = de * g0 * f1 * (1 - f1)
	dsf2 = dh * e * f2 * (1 - f2)
	dsg0 = de * f1 * (1 - g0*g0)
	'''
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		inp = sets*total*time + _set*total + istart
		W = _set*wsize + wstart
		out = sets*total*time + _set*total + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		_W = Bx*Ay
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		Yx = Bx
		Yy = Ay
	
		for y in range(Yy):
			for x in range(Yx):
				#### f0,f1,f2 = f(x@W + h[-1]@W + b)
				f0,f1,f2,g0 = 0
				####
				# f0 is f0<x,y>
				####
	
				#### .W
				for k in range(Ax):
					vpos = inp + y*Ax + k
					wpos = (k*Bx + y)
					f0 += var[vpos]*w[W + 0*lineW + wpos]
					f1 += var[vpos]*w[W + 1*lineW + wpos]
					f2 += var[vpos]*w[W + 2*lineW + wpos]
					g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .U
				if time > 0:
					for k in range(Yx):
						#out == t
						#out - total*sets == sets*total*(l-1) + _set*total + istart
						vpos = total*sets*(time-1) + _set*total + ystart + (Bx*Ay) + (y*Ax + k) 
						wpos = _W + k*Bx + y
						f0 += var[vpos]*w[W + 0*lineW + wpos]
						f1 += var[vpos]*w[W + 1*lineW + wpos]
						f2 += var[vpos]*w[W + 2*lineW + wpos]
						g0 += var[vpos]*w[W + 3*lineW + wpos]
	
				#### .B
				wpos = _W + _U + y*Bx + x
				f0 += w[W + 0*lineW + wpos]
				f1 += w[W + 1*lineW + wpos]
				f2 += w[W + 2*lineW + wpos]
				g0 += w[W + 3*lineW + wpos]
	
				#### activate( _sum )
				f0 = logistic(f0)
				f1 = logistic(f1)
				f2 = logistic(f2)
				g0 = tanh(g0)
	
				#### e = f0 * e[-1] + f1 * g0
				#### l-1 >= 0
				e_1 = (var[out-total*sets + y] if time-1 >= 0 else 0)
	
				e = f0*e_1 + f1*g0
				h = f2 * e
	
				locd_f0 = f0#f2*e_1*( f0*(1 - f0) )
				locd_f1 = f1#f2*g0*( f1*(1 - f1) )
				locd_f2 = f2#e*( f2*(1 - f2) )
				locd_g0 = g0#f2*f1*( 1 - g0*g0)
	
				var[out + (y*Bx + x)] = e
				var[out + Bx*Ay + (y*Bx + x)] = h
	
				locd[locdpos + 0*Bx*Ay + y] = locd_f0
				locd[locdpos + 1*Bx*Ay + y] = locd_f1
				locd[locdpos + 2*Bx*Ay + y] = locd_f2
				lcod[locdpos + 3*Bx*Ay + y] = locd_g0
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
	
		inp = sets*total*time + _set*total + istart
		W = _set*wsize + wstart
		out = sets*total*time + _set*total + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		_W = Bx*Ay
		_U = Bx*Bx
		_B = Bx*Ay
	
		lineW = _W + _U + _B
	
		Yx = Bx
		Yy = Ay
	
		for y in range(Yy):
			for x in range(Yx):
				epos = out + (y*Bx + x)
				e_1pos = total*sets*(time-1) + total*_set + ystart + (y*Bx + x) #if l == 0 , e_1pos <= 0
				hpos = out + Bx*Ay + (y*Bx + x)	#Bx*Ay is the space of `e`
	
				dH = grad[hpos]
	
				f0 = locd[locdpos + 0*Bx*Ay + y] * dH
				f1 = locd[locdpos + 1*Bx*Ay + y] * dH
				f2 = locd[locdpos + 2*Bx*Ay + y] * dH
				g0 = lcod[locdpos + 3*Bx*Ay + y] * dH
	
				de = grad[epos] + dH * f2		#grad(e) += dH*f2
	
				grad[epos] = de
	
				if time > 0:
					grad[e_1pos] += de * f0
				dsf0 = de * var[e_1pos] * f0 * (1 - f0)
				dsf1 = de * g0 * f1 * (1 - f1)
				dsf2 = dH * e * f2 * (1 - f2)
				dsg0 = de * f1 * (1 - g0*g0)
	
				#### .W
				for k in range(Ax):
					vpos = inp + y*Ax + k
					wpos = (k*Bx + y)
	
					#f0 += var[vpos]*w[W + wpos]
					grad[vpos] += locd_f0 * w[W + 0*lineW + wpos]
					meand[W + 0*lineW + wpos] += locd_f0 * var[vpos]
	
					#f1 += var[vpos]*w[W + lineW + wpos]
					grad[vpos] += locd_f1 * w[W + 1*lineW + wpos]
					meand[W + 1*lineW + wpos] += locd_f1 * var[vpos]
	
					#f2 += var[vpos]*w[W + 2*lineW + wpos]
					grad[vpos] += locd_f2 * w[W + 2*lineW + wpos]
					meand[W + 2*lineW + wpos] += locd_f2 * var[vpos]
	
					#g0 += var[vpos]*w[W + 3*lineW + wpos]
					grad[vpos] += locd_g0 * w[W + 3*lineW + wpos]
					meand[W + 3*lineW + wpos] += locd_g0 * var[vpos]
	
				#### .U
				if time > 0:
					for k in range(Yx):
						#out == t
						#out - total*sets == sets*total*(l-1) + _set*total + istart
						vpos = sets*total*(time-1) + _set*total + ystart + Bx*Ax + (y*Ax + k) 	#h[-1][y][x]
						wpos = _W + (k*Bx + y)
	
						#f0 += var[vpos]*w[W + wpos]
						grad[vpos] += locd_f0 * w[W + 0*lineW + wpos]
						meand[W + 0*lineW + wpos] += locd_f0 * var[vpos]
	
						#f1 += var[vpos]*w[W + lineW + wpos]
						grad[vpos] += locd_f1 * w[W + 1*lineW + wpos]
						meand[W + 1*lineW + wpos] += locd_f1 * var[vpos]
	
						#f2 += var[vpos]*w[W + 2*lineW + wpos]
						grad[vpos] += locd_f2 * w[W + 2*lineW + wpos]
						meand[W + 2*lineW + wpos] += locd_f2 * var[vpos]
	
						#g0 += var[vpos]*w[W + 3*lineW + wpos]
						grad[vpos] += locd_g0 * w[W + 3*lineW + wpos]
						meand[W + 3*lineW + wpos] += locd_g0 * var[vpos]
	
				#### .B
				wpos = _W + _U + (y*Bx + x)
	
				meand[W + 0*lineW + wpos] += dsf0
				meand[W + 1*lineW + wpos] += dsf1
				meand[W + 2*lineW + wpos] += dsf2
				meand[W + 3*lineW + wpos] += dsg0
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return Bx*Ay
	
	def buildstackmodel_weights(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return 4*(Bx*Ax + Bx*Bx + Bx*Ay)
	
	def buildstackmodel_locds(self):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return 4*Bx*Ay
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return [(f'{_id}.e [lstm2d]',stack_start), (f'{_id}.h [lstm2d]',stack_start+Ax*Ay)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		wline = Bx*Ax + Bx*Bx + Bx*Ay
		return [
			(f'{_id}.Wf0 [lstm2d]',stack_start),(f'{_id}.Uf0 [lstm2d]',stack_start+Bx*Ax),(f'{_id}.Bf0 [lstm2d]',stack_start+Bx*Ax+Bx*Bx),
			(f'{_id}.Wf1 [lstm2d]',stack_start+wline),(f'{_id}.Uf1 [lstm2d]',stack_start+wline+Bx*Ax),(f'{_id}.Bf1 [lstm2d]',stack_start+wline+Bx*Ax+Bx*Bx),
			(f'{_id}.Wf2 [lstm2d]',stack_start+2*wline),(f'{_id}.Uf2 [lstm2d]',stack_start+2*wline++Bx*Ax),(f'{_id}.Bf2 [lstm2d]',stack_start+2*wline+Bx*Ax+Bx*Bx),
			(f'{_id}.Wg0 [lstm2d]',stack_start+3*wline),(f'{_id}.Ug0 [lstm2d]',stack_start+3*wline+Bx*Ax),(f'{_id}.Bg0 [lstm2d]',stack_start+3*wline+Bx*Ax+Bx*Bx)]
	
	def labelstackmodel_locds(self,_id, stack_start):
		Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate = list(self.params.values())
		return [(f'{_id}.f0',stack_start),(f'{_id}.f1',stack_start+Bx*Ay),(f'{_id}.f2',stack_start+2*Bx*Ay),(f'{_id}.g0',stack_start+3*Bx*Ay)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_lstm2d = "Ax", "Ay", "Bx", "drate"
	
	requiredposition_lstm2d = 1,1,1,0,0,0,0,1
	
	def setupparamsstackmodel_lstm2d(self, ystart, wstart, lstart, required):
		Ax,Ay,Bx,drate = required
		return Ax,Ay,Bx,istart,ystart,wstart,lstart,drate