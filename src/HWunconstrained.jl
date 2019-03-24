module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra
  using Plots
	using Calculus
  using Optim

    export maximize_like_grad, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10000)
						mu = [0.;0.;0.]
            sig = [1. 0. 0.;0. 1. 0.;0. 0. 1.]
						x = zeros(3,n)
						for i in 1:n
					     x[:,i] =rand(MvNormal(mu,sig))
						end
					  β = [1,1.5,-0.5]
						temp = x'*β
						ϵ = randn(n)
						y = zeros(n)
						y_latent = temp + ϵ
						for i in 1:n
		          if y_latent[i] > 0
				         y[i] = 1
		          else nothing
		          end
            end
						return Dict("β"=>β,"n"=>n,"x"=>x',"y"=>y)
  end



	# log likelihood function at x
	function loglik(beta,d=d)
	   temp = d["x"]*beta
	   Prob = cdf.(Normal(),temp')
	   y = zeros(d["n"])
		 ϵ = randn(d["n"])
		 y_latent = temp + ϵ
		 for i in 1:d["n"]
			 if y_latent[i] > 0
					y[i] = 1
			 else nothing
			 end
		 end
       return Prob*y+(ones(1,d["n"])-Prob)*(ones(d["n"])-y)
	end


  #plotLike to plot the log likelihood function
  function plotLike3(d) # fix the third dimension
  x = y = [i for i=-5:0.5:5];
	temp  = [loglik([i,j,-0.5],d) for i in x for j in y]
	surface(x,y,temp,title="loglikelihood by fixing third beta",fillalpha=0.8,leg=false,fillcolor=:heat)
  end

	#plotLike to plot the log likelihood function
  function plotLike2(d) # fix the second dimension
  x = y = [i for i=-5:0.5:5];
	temp = [loglik([i,1.5,j],d) for i in x for j in y]
	surface(x,y,temp,title="loglikelihood by fixing second beta",fillalpha=0.8,leg=false,fillcolor=:heat)
  end

	#plotLike to plot the log likelihood function
  function plotLike1(d) # fix the first dimension
  x = y = [i for i=-5:0.5:5];
  temp = [loglik([1,i,j],d) for i in x for j in y]
	surface(x,y,temp,title="loglikelihood by fixing first beta",fillalpha=0.8,leg=false,fillcolor=:heat)
  end

	# gradient of the likelihood at x by fixing the first beta
	function grad_plot!(beta,d)
	x = y = [i for i=-5:0.5:5];
  gradient = [Calculus.gradient(x->loglik([beta[1],x[1],x[2]],d),[i,j]) for i in x for j in y]
	gradient_norm = [norm(Calculus.gradient(x->loglik([beta[1],x[1],x[2]],d),[i,j])) for i in x for j in y]
	surface(x,y,gradient_norm,title="norm of the gradient",fillalpha=0.8,leg=false,fillcolor=:heat)
  return gradient
	end

	function grad!(storage,beta,d)
	x = y = [i for i=-5:0.5:5];
	storage = Calculus.gradient(x->loglik(x,d),[i,j,k]) for i in x for j in y for k in z]
	end

	# hessian of the likelihood at x
	function hessian!(storage,betas,d)
	x = y = [i for i=-5:0.5:5];
  storage = [Calculus.hessian(x->loglik(x,d),[i,j,k]) for i in x for j in y for k in z]
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
	function maximize_like(x0=[0.8,1.0,-0.1],d,meth=NelderMead())
  optimize(loglik(x,d),grad!,x0,NelderMead())


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
	function plotLike()



























	end
	function plotGrad()
































	end



	function saveplots()
		p1 = plotLike()
		savefig(p1,joinpath(dirname(@__FILE__),"..","likelihood.png"))
		p2 = plotGrad()
		savefig(p2,joinpath(dirname(@__FILE__),"..","gradient.png"))
		@info("saved both plots.")
	end

end
