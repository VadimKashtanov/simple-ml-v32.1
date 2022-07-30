from math import exp, tanh

'''			lstm1d : [X] -> [Y]
x:[X]

f0 = f(x@W + h[-1]@U + b)
f1 = f(x@W + h[-1]@U + b)
f2 = f(x@W + h[-1]@U + b)
g0 = g(x@W + h[-1]@U + b)
e = f0 * e[-1] + f1 * g0
h = f2 * e

h:[Y]

f(x) = 1 / (1 + exp(-x))
g(x) = tanh(x)


Input = X
Output = Y*2 (store e and h)

W:[X,Y]
U:[X,X]
B:[Y]
'''

'''	Var struct
e:[Y]
h:[Y]
'''

'''	Weight struct
f0W:[X,Y]
f0U:[Y,Y]
f0B:[Y]
f1W:[X,Y]
f1U:[Y,Y]
f1B:[Y]
f2W:[X,Y]
f2U:[Y,Y]
f2B:[Y]
f3W:[X,Y]
f3U:[Y,Y]
f3B:[Y]
'''

'''	Locd struct
f0:[Y]
f1:[Y]
f2:[Y]
g0:[Y]
e1:[Y]	//f2*f0
'''

#	Params : [X,Y, istart,ystart,wstart,locdstart, drate]

logistic = lambda x: 1 / (1 + exp(-x))

class LSTM1D(Inst):
	_id = 4
	ID = 4

	name='LSTM1D'
	params_names=['X', 'Y', 'istart', 'ystart', 'wstart', 'locdstart', 'drate']

	def check(self):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return all(i >= 0 for i in params) and X > 0 and Y > 0
	
	def mdl(self,
		total:int, l:int,
		var:[float], w:[float]):
	
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		inp = total*l + istart
		W = wstart
		out = total*l + ystart
	
		lineW = X*Y + Y*Y + Y
	
		for y in range(Y):
			#### f0,f1,f2 = f(x@W + h[-1]@W + b)
			f0,f1,f2,g0 = 0
	
			#### x@.W
			for k in range(X):
				vpos = inp + k
				wpos = k*X + y
				f0 += var[vpos]*w[W + wpos]
				f1 += var[vpos]*w[W + lineW + wpos]
				f2 += var[vpos]*w[W + 2*lineW + wpos]
				g0 += var[vpos]*w[W + 3*lineW + wpos]
	
			#### h[-1]@.U
			if l > 0:
				for k in range(Y):
					#ystart car on va chercher h[-1]
					vpos = total*(l-1) + ystart + Y + k 		# + Y  car output == e[Y] + [Y]
					wpos = X*Y + k*X + y
					f0 += var[vpos]*w[W + wpos]
					f1 += var[vpos]*w[W + lineW + wpos]
					f2 += var[vpos]*w[W + 2*lineW + wpos]
					g0 += var[vpos]*w[W + 3*lineW + wpos]
			
			#### .B
			wpos = X*Y + Y*Y + y
			f0 += w[W + wpos]
			f1 += w[W + lineW + wpos]
			f2 += w[W + 2*lineW + wpos]
			g0 += w[W + 3*lineW + wpos]
	
			#### activate(_sum)
			f0 = logistic(f0)
			f1 = logistic(f1)
			f2 = logistic(f2)
			g0 = tanh(g0)
	
			##### e = f0 * e[-1] + f1 * g0
			##### l - 1 have to be >= 0     <=> (l>=1) <=> (l>0)
			e_1 = (var[out-total + y] if l-1 >= 0 else 0)
	
			e = f0*e_1 + f1*g0
			h = f2 * (e)	#f(x)=x
	
			var[out + y] = e
			var[out + Y + y] = h
	
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
	
	============== Locd ====================
	
	locd_f0 = f0
	locd_f1 = f1
	locd_f2 = f2
	locd_g0 = g0
	
	otherwise we would have to store also : f2 (for de), f0 (for grad(e[-1]))
	'''
	
	def forward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float]):
		
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		inp = total*sets*time + total*_set + istart
		W = ws*_set + wstart
		out = total*sets*time + total*_set + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		lineW = X*Y + Y*Y + Y
	
		for y in range(Y):
	
			#### f0,f1,f2 = logistic(x@W + h[-1]@U + B)
			#### g0 	  = tanh 	(x@W + h[-1]@U + B)
			f0,f1,f2,g0 = 0
	
			#### .W
			for k in range(X):
				vpos = inp + k
				wpos = k*X + y
				f0 += var[vpos]*w[W + 0*wpos]
				f1 += var[vpos]*w[W + 1*lineW + wpos]
				f2 += var[vpos]*w[W + 2*lineW + wpos]
				g0 += var[vpos]*w[W + 3*lineW + wpos]
	
			#### .U
			if time > 0:
				for k in range(Y):
					vpos = total*sets*(time-1) + total*_set + ystart + Y + k
					wpos = X*Y + k*X + y
					f0 += var[vpos]*w[W + 0*lineW + wpos]
					f1 += var[vpos]*w[W + 1*lineW + wpos]
					f2 += var[vpos]*w[W + 2*lineW + wpos]
					g0 += var[vpos]*w[W + 3*lineW + wpos]
	
			#### .B
			wpos = X*Y + Y*Y + y
			f0 += w[W + 0*lineW + wpos]
			f1 += w[W + 1*lineW + wpos]
			f2 += w[W + 2*lineW + wpos]
			g0 += w[W + 3*lineW + wpos]
	
			#### activ(_sum)
			f0 = logistic(f0)
			f1 = logistic(f1)
			f2 = logistic(f2)
			g0 = tanh(g0)
	
			##### e = f0 * e[-1] + f1 * g0
			##### l - 1 have to be >= 0
			e_1 = (var[total*sets*(time-1) + total*_set + ystart + y] if time-1 >= 0 else 0)
	
			e = f0*e_1 + f1*g0;
			h = f2 * e
	
			locd_f0 = f0#f2*e_1*( f0*(1 - f0) )
			locd_f1 = f1#f2*g0*( f1*(1 - f1) )
			locd_f2 = f2#e*( f2*(1 - f2) )
			locd_g0 = g0#f2*f1*( 1 - g0*g0)
	
			var[out + y] = e
			var[out + Y + y] = h
	
			locd[locdpos + 0*Y + y] = locd_f0
			locd[locdpos + 1*Y + y] = locd_f1
			locd[locdpos + 2*Y + y] = locd_f2
			locd[locdpos + 3*Y + y] = locd_g0
	
	def backward(self,
		sets:int, total:int, ws:int, locds:int, _set:int, time:int, 
		w:[float], var:[float], locd:[float], grad:[float], meand:[float]):
		
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
	
		inp = total*sets*time + total*_set + istart
		W = ws*_set + wstart
		out = total*sets*time + total*_set + ystart
		locdpos = locds*sets*time + locds*_set + locdstart
	
		lineW = X*Y + Y*Y + Y
	
		for y in range(Y):
	
			epos = out + y
			e_1pos = total*sets*(time-1) + total*_set + ystart + y #if l == 0 , e_1pos <= 0
			hpos = out + Y + y
			#e == [out + y]
			#h == [out + Y + y]
			dH = grad[hpos]
	
			f0 = locd[locdpos + 0*Y + y] * dH
			f1 = locd[locdpos + 1*Y + y] * dH
			f2 = locd[locdpos + 2*Y + y] * dH
			g0 = lcod[locdpos + 3*Y + y] * dH
	
			de = grad[epos] + dH * f2		#grad(e) += dH*f2
	
			grad[epos] = de
	
			if time > 0:
				grad[e_1pos] += de * f0
			dsf0 = de * var[e_1pos] * f0 * (1 - f0)
			dsf1 = de * g0 * f1 * (1 - f1)
			dsf2 = dH * e * f2 * (1 - f2)
			dsg0 = de * f1 * (1 - g0*g0)
	
			for k in range(X):
				wpos = k*X + k
				vpos = inp + k
	
				#f0 += var[inp+x]*w[W + k*X + x]
				grad[vpos] += dsf0 * w[W + 0*lineW + wpos]
				meand[W + 0*lineW + wpos] += locd_f0 * var[vpos]
	
				#f1 += var[inp+x]*w[W + lineW + y*X + x]
				grad[vpos] += dsf1 * w[W + 1*lineW + wpos]
				meand[W + 1*lineW + wpos] += dsf1 * var[vpos]
	
				#f2 += var[inp+x]*w[W + 2*lineW + y*X + x]
				grad[vpos] += dsf2 * w[W + 2*lineW + wpos]
				meand[W + 2*lineW + wpos] += dsf2 * var[vpos]
	
				#g0 += var[inp+x]*w[W + 3*lineW + y*X + x]
				grad[vpos] += dsg0 * w[W + 3*lineW + wpos]
				meand[W + 3*lineW + wpos] += dsg0 * var[vpos]
	
			if time > 0:
				for k in range(Y):
					wpos = (X*Y) + (k*X + y)
					vpos = total*sets*(time-1) + total*_set + ystart + Y + k
	
					#f0 += var[vpos]*w[W + X*Y + y*Y + _y]
					grad[vpos] += dsf0 * w[W + 0*lineW + wpos]
					meand[W + 0*lineW + wpos] += dsf0 * var[vpos]
	
					#f1 += var[vpos]*w[W + lineW + X*Y + y*Y + _y]
					grad[vpos] += dsf1 * w[W + 1*lineW + wpos]
					meand[W + 1*lineW + wpos] += dsf1 * var[vpos]
	
					#f2 += var[vpos]*w[W + 2*lineW + X*Y + y*Y + _y]
					grad[vpos] += dsf2 * w[W + 2*lineW + wpos]
					meand[W + 2*lineW + wpos] += dsf2 * var[vpos]
	
					#g0 += var[vpos]*w[W + 3*lineW + X*Y + y*Y + _y]
					grad[vpos] += dsg0 * w[W + 3*lineW + wpos]
					meand[W + 3*lineW + wpos] += dsg0 * var[vpos]
	
			wpos = X*Y + Y*Y + y
			#f0 += w[W + X*Y + Y*Y + y]
			meand[W + 0*lineW + wpos] += dsf0
	
			#f1 += w[W + lineW + X*Y + Y*Y + y]
			meand[W + 1*lineW + wpos] += dsf1
	
			#f2 += w[W + 2*lineW + X*Y + Y*Y + y]
			meand[W + 2*lineW + wpos] += dsf2
	
			#g0 += w[W + 3*lineW + X*Y + Y*Y + y]
			meand[W + 3*lineW + wpos] += dsg0
	
	####################### Spetial functions for applications ##########################
	
	
	
	#### Build Stack Model  (Applications : "stack_model.py", )
	
	def buildstackmodel_vars(self):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return 2*Y 		#we store `h` and `e` because `e` is for `e[-1]` and `h` is real output
	
	def buildstackmodel_weights(self):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return 4*(X*Y + Y*Y + Y)
	
	def buildstackmodel_locds(self):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return 4*(Y)
	
	#### Labels Stack Model  (Applications : "stack_model.py", )
	
	def labelstackmodel_vars(self, _id, stack_start):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.e [lstm1d]',stack_start), (f'{_id}.h [lstm1d]',stack_start + X)]
	
	def labelstackmodel_weights(self,_id, stack_start):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		wline = X*Y + Y*Y + Y
		return [
			(f'{_id}.Wf0 [lstm1d]',stack_start),(f'{_id}.Uf0 [lstm1d]',stack_start + X*Y),(f'{_id}.Bf0 [lstm1d]',stack_start + X*Y+Y*Y),
			(f'{_id}.Wf1 [lstm1d]',stack_start + wline),(f'{_id}.Uf1 [lstm1d]',stack_start + wline+X*Y),(f'{_id}.Bf1 [lstm1d]',stack_start + wline+X*Y+Y*Y),
			(f'{_id}.Wf2 [lstm1d]',stack_start + 2*wline),(f'{_id}.Uf2 [lstm1d]',stack_start + 2*wline+X*Y),(f'{_id}.Bf2 [lstm1d]',stack_start + 2*wline+X*Y+Y*Y),
			(f'{_id}.Wg0 [lstm1d]',stack_start + 3*wline),(f'{_id}.Ug0 [lstm1d]',stack_start + 3*wline+X*Y),(f'{_id}.Bg0 [lstm1d]',stack_start + 3*wline+X*Y+Y*Y)
		]
	
	def labelstackmodel_locds(self,_id, stack_start):
		X,Y, istart,ystart,wstart,locdstart, drop_rate = list(self.params.values())
		return [(f'{_id}.f0',stack_start),(f'{_id}.f1',stack_start+Y),(f'{_id}.f2',stack_start+2*Y),(f'{_id}.g0',stack_start+3*Y)]
	
	### Setput Params Stack Model
	
	requiredforsetupparams_lstm1d = "X", "Y", "drate"
	
	requiredposition_lstm1d = 1,1,0,0,0,0,1
	
	def setupparamsstackmodel_lstm1d(self, ystart, wstart, lstart, required):
		X,Y,drate = required
		return X,Y,istart,ystart,wstart,lstart,drate