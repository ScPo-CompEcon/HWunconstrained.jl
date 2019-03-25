using Test
using GLM
using Statistics
using DataFrames
using Calculus


@testset "HWunconstrained.jl" begin


@testset "basics" begin

	@testset "Test Data Construction" begin

		# create data
		d = makeData(18)

		# test dataframe has 18 rows
		@test size(d["X"],1) == 18

		# test generated y vector has a mean less than 1
	end

	@testset "Test Return value of likelihood" begin

		@test size(loglik(d["beta"],makeData()),1) == 1

	end

	@testset "Test return value of gradient is nothing" begin
		# gradient should not return anything,
		# but modify a vector in place.
		d = makeData()
		G = zeros(d["n"])
		beta = d["beta"]
		@test HWunconstrained.grad!(G, beta, d) == nothing

	end


	@testset "gradient vs finite difference" begin
		d = makeData()
		G = zeros(d["n"])
		beta = d["beta"]
		HWunconstrained.grad!(G,beta, d)
		F = Calculus.finite_difference( x-> loglik(x,d), d["beta"],:central)
		@test G ≈ F

	end
end

@testset "test maximization results" begin

	ttol = 2e-1

	@testset "maximize returns approximate result" begin
		m = HWunconstrained.maximize_like();
		d = makeData();
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "maximize_grad returns accurate result" begin
		m = HWunconstrained.maximize_like_grad();
		d = makeData();
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "maximize_grad_hess returns accurate result" begin
		m = HWunconstrained.maximize_like_grad_hess();
		d = makeData();
		@test maximum(abs,m.minimizer .- d["beta"]) < ttol
	end

	@testset "gradient is close to zero at max like estimate" begin
		m = HWunconstrained.maximize_like_grad();
		d = makeData()
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
