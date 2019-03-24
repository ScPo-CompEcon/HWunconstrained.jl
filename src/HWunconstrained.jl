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
		Random.seed!(1)
		N = n
		norm = Normal()
		beta = [1, 1.5, -0.5]
		X = [randn(N) randn(N) randn(N)]
		ϵ = rand(norm, N)
		y_latent = X*beta + ϵ

# Define the binary response variable (if latent variable has positive value)
		y = zeros(N)
		for i in eachindex(y_latent)
			if y_latent[i] > 0
			y[i] = 1
			end
		end

		return Dict("beta"=>beta,"n"=>N,"X"=>X,"y"=>y,"dist"=>norm)
	end


	# log likelihood function at x
function loglik(betas::Vector,d::Dict)

	l_i = zeros(length(d["y"]))
	Φ = zeros(length(d["y"]))
	for i in eachindex(d["y"])
	Φ[i] = cdf(d["dist"], d["X"][i,:]' * betas)
	l_i[i] = Φ[i]^(d["y"][i]) * (1 - Φ[i])^(1 - d["y"][i])
	end

	log_lik = -(1/d["n"]) * sum(log.(l_i))
	return log_lik

end


	# gradient of the likelihood at x
function grad!(storage::Vector,betas::Vector,d)
		ϕ = zeros(length(d["y"]))
		Φ = zeros(length(d["y"]))
		f = zeros(length(d["y"]), length(betas))

	for j in eachindex(d["y"])
		ϕ[j] = pdf(d["dist"], d["X"][j,:]' * betas)
		Φ[j] = cdf(d["dist"], d["X"][j,:]' * betas)
		f[j,:] = ((d["y"][j] * ϕ[j]/Φ[j] - ((1 - d["y"][j]) * ϕ[j]/(1 - Φ[j]))) * d["X"][j,:]') * (1/d["n"])
	end

	for col in 1:3
		storage[col] = sum(f[:,col])
	end
	return storage
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
function maximize_like(x0=[0.8,1.0,-0.1],meth=NelderMead())
res = optimize(arg->loglik(arg, d), x0, meth)
return res.minimizer
end


function maximize_like_helpNM(x0=[ 1; 1.5; -0.5 ],meth=NelderMead())
	res = optimize(arg -> loglik(arg, d), x0, meth)
	return res.minimizer
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=BFGS())
	res = optimize(arg->loglik(arg, d), x0, meth)
	return res
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


function plotLike()
	g = collect(-3:0.1:3)
	h1 = ones(length(g))
	h2 = repeat([1.5], outer = length(g))
	h3 = repeat([-0.5], outer = length(g))

	A1 = [g h2 h3]
	A2 = [h1 g h3]
	A3 = [h1 h2 g]
	A = [A1, A2, A3]

L = [zeros(length(g)) zeros(length(g)) zeros(length(g)) g]
for j in 1:3
	for i in 1:length(g)
		L[i,j] = loglik(A[j][i,:], d)
	end
end

plot(L[:,4], [L[:,1] L[:,2] L[:,3]], layout = 3, ylabel = "Likelihood", label = ["L(beta1)", "L(beta2)", "L(beta3)"])

end



function plotGrad()
	g = collect(-3:0.1:3)
	h1 = ones(length(g))
	h2 = repeat([1.5], outer = length(g))
	h3 = repeat([-0.5], outer = length(g))

	A1 = [g h2 h3]
	A2 = [h1 g h3]
	A3 = [h1 h2 g]
	A = [A1, A2, A3]

	G = [zeros(length(g)) zeros(length(g)) zeros(length(g)) g]
	for j in 1:3
		for i in 1:length(g)
			G[i,j] = grad!(zeros(3), A[j][i,:], d)[j]
			# only plots the partial derivative with respect to the element j of betas, not with respect to the other two.
		end
	end

	plot(G[:,4], [G[:,1] G[:,2] G[:,3]], layout = 3, ylabel = "Partial derivative", label = ["grad(beta1)", "grad(beta2)", "grad(beta3)"])

	end


function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
end
