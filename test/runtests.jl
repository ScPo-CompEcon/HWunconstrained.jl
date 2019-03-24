using Test
using GLM
using Statistics
using DataFrames
using Calculus
using HWunconstrained
using ForwardDiff


@testset "HWunconstrained.jl" begin


@testset "basics" begin

	@testset "Test Data Construction" begin

		# create data
		d = makeData(18)

		# test dataframe has 18 rows
		@test size(d["X"], 1) == size(d["y"], 1)

		# test generated y vector has a mean less than 1
		@test mean(d["y"]) < 1
	end

	@testset "Test Return value of likelihood" begin
		d = makeData()
		@test loglik([randn(), randn(), randn()], d) < 0
	end

	@testset "Test return value of gradient is nothing" begin
		# gradient should not return anything,
		# but modify a vector in place.
			d = makeData()
			G = zeros(3)
			@test grad!(G, [1., 1., 1.], d) == nothing
			@test G != zeros(3)
	end


	@testset "gradient vs finite difference" begin
		G = zeros(3)
		d = makeData()
		pts = [1., 1., 1.]
		grad!(G, pts, d)
		@test 1 == 1
		#@test G \approx ForwardDiff.gradient(x->loglik(x, a), pts)
	end
end

@testset "test maximization results" begin

	ttol = 2e-1
	d = makeData()

	@testset "maximize returns approximate result" begin
		m = HWunconstrained.maximize_like(loglik, d);
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "maximize_grad returns accurate result" begin
		m = HWunconstrained.maximize_like_grad(loglik, grad!, d);
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "maximize_grad_hess returns accurate result" begin
		m = HWunconstrained.maximize_like_grad_hess();
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "gradient is close to zero at max like estimate" begin
		m = HWunconstrained.maximize_like_grad(loglik, grad!, d);
		gradvec = ones(length(d["beta"]))
		r = HWunconstrained.grad!(gradvec,m.minimizer,d)

		@test r == nothing 

		@test maximum(abs,gradvec) < 1e-6

	end

end

@testset "test against GLM" begin

	@testset "estimates vs GLM" begin


	end

	@testset "standard errors vs GLM" begin


	end

end

    
end
