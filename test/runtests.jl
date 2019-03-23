using Test
using GLM
using Statistics
using DataFrames
using Calculus
using HWunconstrained


@testset "HWunconstrained.jl" begin


	@testset "basics" begin

		@testset "Test Data Construction" begin

			# create data
			d = HWunconstrained.makeData(18);

			# test dataframe has 18 rows
			@test size(d["X"],1) == 18
			@test size(d["y"],1) == 18
			# test generated y vector has a mean less than 1
			@test Statistics.mean(d["y"]) < 1
		end

		@testset "Test Return value of likelihood" begin

		end

		@testset "Test return value of gradient is nothing" begin
			# gradient should not return anything,
			d = HWunconstrained.makeData(10);
			G = zeros(3);
			@test isnothing(HWunconstrained.grad!(G,[1.2,1.0,0.1],d))
			# but modify a vector in place.
			G = zeros(3);
			@test isequal(G, HWunconstrained.grad!(G,[1.2,1.0,0.1],d)) == false
		end


		@testset "gradient vs finite difference" begin
		end
	end

	@testset "test maximization results" begin

		ttol = 2e-1

		@testset "maximize returns approximate result" begin
			m = HWunconstrained.maximize_like();
			d = HWunconstrained.makeData();
			@test maximum(abs,m.minimizer .- d["beta"]) < ttol
		end

		@testset "maximize_grad returns accurate result" begin
			m = HWunconstrained.maximize_like_grad();
			d = HWunconstrained.makeData();
			@test maximum(abs,m.minimizer .- d["beta"]) < ttol
		end

		@testset "maximize_grad_hess returns accurate result" begin
			m = HWunconstrained.maximize_like_grad_hess();
			d = HWunconstrained.makeData();
			@test maximum(abs,m.minimizer .- d["beta"]) < ttol
		end

		@testset "gradient is close to zero at max like estimate" begin
			m = HWunconstrained.maximize_like_grad();
			d = HWunconstrained.makeData()
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
