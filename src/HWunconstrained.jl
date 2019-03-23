
module HWunconstrained


	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra
	using Plots.PlotMeasures



    export maximize_like_grad, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10_000)
		#beta
		beta = [1,1.5,-0.5]
		#distr
		norm = Normal()
		#setting seed
		Random.seed!(9556)
		#generating X
		X = zeros(n,3)
		for i in 1:n, j in 1:3
			X[i,j] = randn()
		end
		#generating y
		y = zeros(n)
		for i in 1:n
			rand() <= cdf.(norm,(X[i,:]'*beta)) ? y[i] = 1 : nothing
		end
		numobs = n
		return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
	end

	# log likelihood function at x
	function loglik(betas::Vector,d::Dict)
		n = d["n"]
		X = d["X"]
		y = d["y"]
		return sum(y'*log.(cdf.(Normal(),X*betas)) + (ones(n) - y)'*log.(ones(n) - cdf.(Normal(),(X*betas))))/n
	end

	# gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)
		n = d["n"]
		X = d["X"]
		y = d["y"]
		s = zeros(length(betas))
		for j in 1:length(betas), i in 1:n
			s[j] += -((y[i] - cdf.(Normal(),X[i,:]'*betas))/(cdf.(Normal(),X[i,:]'*betas) * (1 - cdf.(Normal(),X[i,:]'*betas))))*pdf.(Normal(),X[i,:]'*betas)*X[i,j]/n
		end
		for i in 1:length(s)
			storage[i] = s[i]
		end
	end


	function grad(betas::Vector,d = makeData())
		n = d["n"]
		X = d["X"]
		y = d["y"]
		s = zeros(length(betas))
		for j in 1:length(s), i in 1:n
			s[j] += ((y[i] - cdf.(Normal(),X[i,:]'*betas))/(cdf.(Normal(),X[i,:]'*betas) * (1 - cdf.(Normal(),X[i,:]'*betas))))*pdf.(Normal(),X[i,:]'*betas)*X[i,j]
		end
		return s
	end


	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)
		n = d["n"]
		X = d["X"]
		y = d["y"]
		Phi(x) = cdf.(Normal(),x)
		Phi_p(x) = pdf.(Normal(),x)
		Phi_pp(x) = (-x)*pdf.(Normal(),x)
		s = zeros(length(betas),length(betas))
		for i in 1:length(betas), j in 1:length(betas), l in 1:n
			phi = Phi(X[l,:]'*betas)
			phip = Phi_p(X[l,:]'*betas)
			phipp = Phi_pp(X[l,:]'*betas)
			s[i,j] += -((((y[l] - phi)*(phipp*(1 - phi)*phi - (phip^2)*(1 - 2*phi))) - (phip^2)*(1 - phi)*phi)/((phi - phi^2)^2))*X[l,i]*X[l,j]/n
		end
		for i in 1:length(betas), j in 1:length(betas)
			storage[i,j] = s[i,j]
		end
	end


	function info_mat(betas::Vector,d)


	end

	function inv_Info(betas::Vector,d)

	end

	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)



	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)

	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(d = makeData(), x0=[0.8,1.0,-0.1], meth=NelderMead())
		minNegloglikE(beta; d = d) = -loglik(beta,d)
		nm = optimize(minNegloglikE, x0, meth)
		return nm
	end

	function maximize_like_helpNM(d = makeData(), x0=[0.8,1.0,-0.1], meth=NelderMead())
		minNegloglikE(beta; d = d) = -loglik(beta,d)
		nm = optimize(minNegloglikE, x0, meth)
		return nm
	end

	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(d = makeData(),x0=[0.8,1.0,-0.1],meth=BFGS())
		minNegloglikE(beta; d = d) = -loglik(beta,d)
		bfgs = optimize(minNegloglikE, (storage,betas)->grad!(storage,betas,d), x0, meth)
		return bfgs
	end

	function maximize_like_grad_helpBFGS(d = makeData(),x0=[0.8,1.0,-0.1],meth=BFGS())
		minNegloglikE(beta; d = d) = -loglik(beta,d)
		bfgs = optimize(minNegloglikE, (storage,betas)->grad!(storage,betas,d), x0, meth)
		return bfgs
	end


	function maximize_like_grad_hess(d = makeData(),x0=[0.8,1.0,-0.1],meth=Newton())
		minNegloglikE(beta; d = d) = -loglik(beta,d)
		new = optimize(minNegloglikE, (storage,betas)->grad!(storage,betas,d),(storage,betas)->hessian!(storage,betas,d), x0, meth)
		return new
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())

	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value
	function plotLike(d = makeData())
		#beta
		beta = [1,1.5,-0.5]
		#beta length
		lb = length(beta)
		#grid length
		lgr = length(Vector(-0.5:0.01:0.5))
		#beta-grid(s) and corresponding loglikelihoods
		B = ones(lgr,lb,lb)
		LB = zeros(lgr,lb)
		for i in 1:lb, j in 1:lgr
			B[j,:,i] = beta
			B[:,i,i] = Vector((beta[i]-0.5):0.01:(beta[i] + 0.5))
			LB[j,i] = loglik(B[j,:,i],d)
		end
		p = plot(layout = lb, dpi = 300)
		for i in 1:lb
			plot!(p[i],B[:,i,i],LB[:,i], legend = false, xlabel = "beta$(i)", ylabel = "Avg Log-likelihood",title= "Changing beta$(i)", titlefontsize = 12, guidefontsize = 9)
		end
		return p
	end

	function plotGrad(d = makeData())
		#beta
		beta = [1,1.5,-0.5]
		#beta length
		lb = length(beta)
		#grid length
		lgr = length(Vector(-0.5:0.01:0.5))
		#beta-grid(s) and corresponding gradients of loglikelihoods
		B = ones(lgr,lb,lb)
		GB = zeros(lgr,lb,lb)
		for i in 1:lb, j in 1:lgr
			B[j,:,i] = beta
			B[:,i,i] = Vector((beta[i]-0.5):0.01:(beta[i] + 0.5))
			GB[j,:,i] = grad(B[j,:,i],d)
		end
		#plot
		p = plot(layout = lb*lb, dpi = 300)
		for i in 1:lb, j in 1:lb
			plot!(p[i,j],B[:,i,i],GB[:,j,i], legend = false, xlabel = "beta$(i)", ylabel = "Grad[$(j)] of LogL",title= "Changing beta$(i)", titlefontsize = 12, guidefontsize = 9)
		end
		return p
	end


	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
