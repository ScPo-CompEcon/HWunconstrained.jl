module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra


    export maximize_like_grad, makeData, loglik, plotLike, grad!, plotGrad, maximize_like, maximize_like_grad



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10_000)
		numobs = n
		beta = [1, 1.5, -0.5]
		X = randn(n, length(beta))
		eps = randn(n)
		prob = X * beta + eps
		y = [if i > 0 1 else 0 end for i in prob]
		return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>Normal())
	end


	# log likelihood function at x
	function loglik(betas::Vector,d::Dict)
		prob = cdf.(d["dist"], d["X"] * betas)
		t = dot(d["y"], log.(prob))
		f = dot(1 .- d["y"], log.(1 .- prob))
		return t + f
		end

	# gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)
		Xb = d["X"] * betas
		t = d["y"] .* pdf.(d["dist"], Xb) ./ cdf.(d["dist"], Xb)
		f = (1 .- d["y"]) .* pdf.(d["dist"], Xb) ./ (1 .- cdf.(d["dist"], Xb))
		grad = sum((t .- f) .* d["X"], dims=1)
		storage[:] = grad
		return nothing
	end

	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)
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
	function maximize_like(loglik::Function, d::Dict, x0=[0.8,1.0,-0.1],meth=NelderMead())
		r = optimize(x->-loglik(x, d), x0, method=meth)
		return r
	end

	function maximize_like_helpNM(x0=[ 1; 1.5; -0.5 ],meth=NelderMead())
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(loglik::Function, grad!::Function, d::Dict, x0=[0.8,1.0,-0.1],meth=BFGS())

		r = optimize(x->-loglik(x, d), (g, x)->grad!(g, x, d), x0, method=NelderMead())
		return r
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=Newton())
                     
                              
                                                
                                                                                                                                                           
            
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())
                     
                                      
                                                                                                                                 
                           
                                                                                                      
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value
	function plotLike(d::Dict, loglik::Function, step=0.01, start=-1, stop=2)
		grid = collect(start:step:stop)
		beta = d["beta"]
		len = length(beta)
		ll = zeros(length(grid), len)
		inp = [vcat(beta[1:i-1], vcat(j, beta[i+1:end])) for i in 1:len for j in grid]
		ll = loglik.(inp, Ref(d))
		plot(grid, reshape(ll, (length(grid), len)), layout=(len, 1))
	end

	function plotGrad(d::Dict, loglik::Function, step=0.01, start=-1, stop=2)
		grid = collect(start:step:stop)
		beta = d["beta"]
		len = length(beta)
		ll = zeros(length(grid), len)
		G = [zeros(len) for _ in 1:length(grid) for _ in 1:len]
		inp = [vcat(beta[1:i-1], vcat(j, beta[i+1:end])) for i in 1:len for j in grid]
		grad!.(G, inp, Ref(d))
		G = reshape(vcat(G'...), (length(grid), len, len))
		for i in 1:len
			ll[:, i] = G[:, i, i]
		end
		plot(grid, ll, layout=(len, 1))
	end

	

	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end

