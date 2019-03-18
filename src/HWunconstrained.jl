module HWunconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
	using Random
	using Statistics
	using LinearAlgebra


    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10_000)
                         
                     
 		# your turn!           
                              
                                                    
                         
                        
                    
                                                                                     
		return Dict("beta"=>beta,"n"=>numobs,"X"=>X,"y"=>y,"dist"=>norm)
	end


	# log likelihood function at x
	function loglik(betas::Vector,d::Dict)
                                  
                                           
                                                                                     
                            
                  

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

