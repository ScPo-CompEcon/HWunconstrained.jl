using HWunconstrained
using Test

@testset "HWunconstrained.jl" begin

@testset "basics" begin

	@testset "Test Data Construction" begin

		d = makeData(18)
		# test the the constructed dataframe has 18 rows

		# test that the generated response vector y has a mean that is smaller than 1 ( not all responses are equal 1!)
	end

	@testset "Test that likelihood returns a real number" begin


	end

	@testset "Test gradient returns nothing" begin
		# gradient should not return anything,
		# but modify a vector in place.

	end


	@testset "test gradient vs finite difference approximation" begin

	end
end

@testset "test maximization results" begin

	ttol = 2e-1  # test tolerance

	@testset "maximize returns approximate result" begin

	end

	@testset "maximize_grad returns accurate result" begin

	end

	@testset "maximize_grad_hess returns accurate result" begin
                                                
                 
                                                    
	end

	@testset "gradient is close to zero at max like estimate" begin
                     

                                   

	end

end

@testset "test estimates and std errors against GLM" begin
	
	# generate GLM estimates from our data

	@testset "estimates vs GLM" begin


	end

	@testset "standard errors vs GLM" begin


	end

end

    
end
