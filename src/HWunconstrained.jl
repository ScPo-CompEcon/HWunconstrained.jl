module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra


    export maximize_like_grad, makeData, loglik, grad!, maximize_like, plotLike, plotGrad



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
	           epsilon = randn(numobs)
	           Y = X * beta + epsilon
	           y = 1.0 * (Y .> 0)
	           norm = Normal(0,1)    # create a normal distribution object with mean 0 and variance 1
	           return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
	       end

	function loglik(betas::Vector, d::Dict)
		lh = log.(cdf.(d["dist"],d["X"]*betas) .^ d["y"] .* (1 .- cdf.(d["dist"],d["X"]*betas)) .^ (1 .- d["y"]))
		return (-1) * sum(lh) / d["n"]
	end


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

		for j=1:3
			temp = zeros(length(d["y"]))
 			for i=1:d["n"]
				temp[i] = (d["y"][i] - cdf.(d["dist"], d["X"][i,:]' * betas)) * d["X"][i,j]
			end
			storage[j] = sum(temp)
		end
	end


	# hessian of the likelihood at x
	function hessian!(storage::Matrix,betas::Vector,d)

		#call the gradient we just made

		# Compute the hessian
		.*(storage, storage')
		#change it to matrix







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

# Check i( is intentional that the maximization is not so good)
	function maximize_like(x0=[0.8, 1.0, -1.0],meth=NelderMead())
		maximum_like(betas) = loglik(betas, makeData())
		#maximum_like(betas) = (-1) * sum(log.(cdf.(makeData()["dist"], (makeData()["X"] * betas)) ^makeData()["y"]) .+
		#		(log.(1 .- cdf.(makeData()["dist"], (makeData()["X"] * betas)) ^(1 .- makeData()["y"]))))
		nm = optimize(maximum_like, x0, meth)
		return(nm)
	end
maximize_like()



	function maximize_like_helpNM(x0=[ 1; 1.5; -0.5 ],meth=NelderMead())



	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=BFGS())

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

		# You want to call the loglik function for different vectors of beta and keep loglik in mind
		# increments = 0.01
		totest_values = collect(-3.0:0.01:3.0)
		#for the first element
		j = zeros(length(totest_values))

		d = makeData()
		b1 = d["beta"][1]
		b2 = d["beta"][2]
		b3 = d["beta"][3]
		for i=1:length(totest_values)
			betas_calc = [totest_values[i], b2, b3]
			j[i] = loglik(betas_calc, d)
		end
		p1 = plot(totest_values, j,
			lab="B2 = 1.5 ; B3 = -0.5",
			xlabel = "B1")

		#for the second element
		j = zeros(length(totest_values))
		for i=1:length(totest_values)
			j[i] = loglik([makeData()["beta"][1], totest_values[i], makeData()["beta"][3]],
					makeData())
		end
		p2 = plot(totest_values, j,
			lab="B1 = 1.0 ; B3 = -0.5",
			xlabel = "B2", ylabel = "Log Likelihood function")


		#for the second element
		j = zeros(length(totest_values))
		for i=1:length(totest_values)
			j[i] = loglik([makeData()["beta"][1], makeData()["beta"][2], totest_values[i]],
					makeData())
		end
		p3 = plot(totest_values, j,
			lab="B1 = 1.0 ; B2 = 1.5",
			xlabel = "B3")

		# Plot final graph
		l = @layout [a; b; c]
		return(plot(p1, p2, p3, layout = l))
	end





	function plotGrad()
			# You want to call the loglik function for different vectors of beta and keep loglik in mind
			# increments = 0.01
			totest_values = collect(-2.0:0.01:2.0)

# Element 1
		# reinitialize storage to true values
			betas = makeData()["beta"]
			storage = zeros(3)
			j = zeros(length(totest_values))
		# run the function grad!() to get the gradient & extract the coef of interest
			for i=1:length(totest_values)
				grad!(storage, [totest_values[i], betas[2], betas[3]], makeData())
				j[i] = storage[1]
			end
			p1 = plot(totest_values, j,
				lab="B2 = 1.5 ; B3 = -0.5",
				xlabel = "B1")

# Element 2
			betas = makeData()["beta"]
			storage = zeros(3)
			j = zeros(length(totest_values))
			for i=1:length(totest_values)
				grad!(storage, [betas[1], totest_values[i], betas[3]], makeData())
				j[i] = storage[2]
			end
			p2 = plot(totest_values, j,
				lab="B1 = 1.0 ; B3 = -0.5",
				xlabel = "B2", ylabel = "Log likelihood gradient")

# Element 3
			betas = makeData()["beta"]
			storage = zeros(3)
			j = zeros(length(totest_values))
			for i=1:length(totest_values)
				grad!(storage, [betas[1], betas[2], totest_values[i]], makeData())
				j[i] = storage[3]
			end
			p3 = plot(totest_values, j,
				lab="B1 = 1.0 ; B2 = 1.5",
				xlabel = "B3")
				l = @layout [a; b; c]
			return(plot(p1, p2, p3, layout = l))
		end




	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
