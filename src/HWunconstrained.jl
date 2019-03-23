module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra


    export maximize_like_grad, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.

	function makeData(n=10_000)
        beta = [ 1; 1.5; -0.5 ]
        Random.seed!(54321)
        numobs = n
        X = randn(numobs,3)    # n,k
        # X = hcat(ones(numobs), randn(numobs,2))    # n,k
        epsilon = randn(numobs)
        Y = X * beta + epsilon
        y = 1.0 * (Y .> 0)
        norm = Normal(0,1)    # create a normal distribution object with mean 0 and variance 1
        return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
    end



	# log likelihood function at x
	function loglik(betas::Vector,d::Dict)
		singlelik = d["y"].*log(cdf(d["norm"],d["X"]*betas)) + (1-d["y"]).*log(1-cdf(d["norm"],d["X"]*betas))
		n = size(singlelik,1)
		return sum(singlelik)/n
	end

	# gradient of the likelihood at x
	function grad!(storage::Vector,betas::Vector,d)





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
	function maximize_like(loglik,d,x0=[0.8,1.0,-0.1],meth=NelderMead())

		#result = optimize(arg -> loglik(arg,d), x0, meth)

	end

	function maximize_like_helpNM(d,x0=[ 1; 1.5; -0.5 ],meth=NelderMead())



	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(loglik, d, grad!, x0=[0.8,1.0,-0.1],meth=BFGS())
		#result = optimize(arg -> loglik(arg,d), (g,arg)->grad!(g,arg,d), x0, meth)
	end

	function maximize_like_grad_hess(loglik, d, grad!, hessian!, x0=[0.8,1.0,-0.1],meth=Newton())
		#result = optimize(arg -> loglik(arg,d), (g,arg)->grad!(g,arg,d), (h,arg)->hessian!(g,arg,d), x0, meth)




	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())





	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value
	function plotLike(d::Dict, len=100)
		beta1 = ones(len,1)*transpose(d["beta"])
		beta2 = beta1
		beta3 = beta1
		minval = 0.5
		maxval = 1.5
		factor = reshape(range(minval, maxval, length=len),len, 1)
		beta1[:,1] = beta1[:,1].*factor
		beta2[:,2] = beta2[:,2].*factor
		beta3[:,3] = beta3[:,3].*factor
		for i in eachrow(beta1)
			lik1[i,1] =  loglik(beta1[i,:]::Vector,d::Dict)
			lik2[i,1] =  loglik(beta2[i,:]::Vector,d::Dict)
			lik3[i,1] =  loglik(beta3[i,:]::Vector,d::Dict)
		end
		plot([plot(Lik1,beta1[:,1],label="first parameter"),plot(Lik2,beta2[:,2],label="second parameter"),plot(Lik3,beta3[:,3],label="third parameter")]...)
	end

	function plotGrad(d::Dict, len=100)
		beta1 = ones(len,1)*transpose(d["beta"])
		beta2 = beta1
		beta3 = beta1
		minval = 0.5
		maxval = 1.5
		factor = reshape(range(minval, maxval, length=len),len, 1)
		beta1[:,1] = beta1[:,1].*factor
		beta2[:,2] = beta2[:,2].*factor
		beta3[:,3] = beta3[:,3].*factor
		for i in eachrow(beta1)
			grad1[i,1] = grad!(beta1[i,:]::Vector,d::Dict)
			grad2[i,1] = grad!(beta2[i,:]::Vector,d::Dict)
			grad3[i,1] = grad!(beta3[i,:]::Vector,d::Dict)
		end
		plot([plot(grad1,beta1[:,1],label="first parameter"),plot(grad2,beta2[:,2],label="second parameter"),plot(grad3,beta3[:,3],label="third parameter")]...)
	end



	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
