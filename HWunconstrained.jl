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
    numobs = n
    beta = [1.0, 1.5, -0.5]
    norm = Normal()
    ϵ = randn(n)
    X = [randn(n) randn(n) randn(n)]
    y_latent = X*beta + ϵ
    y = zeros(n)

    for i in 1:n
        if y_latent[i] > 0
            y[i] = 1
        else nothing
        end
    end

    return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
end


	# log likelihood function at x
	function loglik(betas::Vector,d::Dict)
		prob = cdf.(d["dist"], d["X"] * betas)
	    a = d["y"] .* log.(prob)
	    b = (1 .- d["y"]) .* log.(1 .- prob)
	    return a + b
	end

	# gradient of the likelihood at x
	function grad!(s::Vector, betas::Vector, d::Dict)
		Xβ = d["X"] * betas
		a = d["y"] .* pdf.(d["dist"], Xβ) ./ cdf.(d["dist"], Xβ)
		b = (1 .- d["y"]) .* pdf.(d["dist"], Xβ) ./ (1 .- cdf.(d["dist"], Xβ))
		s = sum((a .- b) .* d["X"], dims=1)
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



	end
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

	function plotLike(d::Dict, f::Function, start = -1, step = 0.01, stop = 2)
		    grid = collect(start:step:stop)
		    beta = repeat(transpose(d["beta"]), outer=length(grid))

		    z1 = hcat(beta[:,1:2], grid) #should do all of this in a loop but I wasnt able to manage the step needed for z3
		    z2 = hcat(beta[:,2:3], grid)
		    z3 = hcat(beta[:,1:2:3], grid)
		    input1 = getindex.([z1], 1:size(z1, 1), :) #transforming the matrix into an array of array which is required by loglik
		    input2 = getindex.([z2], 1:size(z2, 1), :)
		    input3 = getindex.([z3], 1:size(z3, 1), :)
		    values1 = zeros(length(grid))
		    values1 = zeros(length(grid))
		    values2 = zeros(length(grid))

		    for i in 1:length(grid)
		        values1[i] = f(input1[i],Ref(d)) #computes log likelihood values
		        values2[i] = f(input2[i],Ref(d))
		        values3[i] = f(input3[i],Ref(d))
		    end

		    values = hcat(values1,values2,values3) #in order to make the graph
		    plot(grid, values, layout=(size(beta, 2), 1))

		end




	function plotGrad(d = makeData(), start = -1, step = 0.01, stop = 2)
		    grid = collect(start:step:stop)
		    beta = repeat(transpose(d["beta"]), outer=length(grid))

		    z1 = hcat(beta[:,1:2], grid) #should do all of this in a loop but I wasnt able to manage to step needed for z3
		    z2 = hcat(beta[:,2:3], grid)
		    z3 = hcat(beta[:,1:2:3], grid)
		    input1 = getindex.([z1], 1:size(z1, 1), :) #transforming the matrix into an array of array so it can pe put into function
		    input2 = getindex.([z2], 1:size(z2, 1), :)
		    input3 = getindex.([z3], 1:size(z3, 1), :)
		    values1 = zeros(length(grid))
		    values1 = zeros(length(grid))
		    values2 = zeros(length(grid))

		    for i in 1:length(grid)
		        values1[i] = grad!(input1[i],d) #computes gradient values
		        values2[i] = grad!(input2[i],d)
		        values3[i] = grad!(input3[i],d)
		    end

		    values = hcat(values1,values2,values3)
		    plot(grid, values, layout=(size(beta, 2), 1))

		end



























	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
