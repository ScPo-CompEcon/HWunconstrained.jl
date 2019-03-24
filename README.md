# HWunconstrained

[![Build Status](https://travis-ci.com/ScPo-CompEcon/HWunconstrained.jl.svg?branch=master)](https://travis-ci.com/ScPo-CompEcon/HWunconstrained.jl)
[![Codecov](https://codecov.io/gh/ScPo-CompEcon/HWunconstrained.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ScPo-CompEcon/HWunconstrained.jl)


## Your Plots

* plot of objective function

![](likelihood.png)


* plot of gradient

![](gradient.png)

## HOW TO DO THIS HOMEWORK

1. click on **fork** top right on this window. After the fork is created
2. in your local julia session, you need to do
    ```julia
    ] dev https://github.com/YOUR_GITHUB_USERNAME/HWunconstrained.jl.git
    ```
3. next, go to the newly created directory to instantiate the package:
    ```julia
    cd(joinpath(DEPOT_PATH[1],"dev","HWunconstrained"))  # go to package location
    ] activate .  
    ] instantiate
    ```
4. Now you can go back to the main environment and start editing the code in your text editor
    ```julia
    ] activate    # no args goes back to main
    using HWunconstrained    # precompiles
    makeData()     # errors: function is incomplete! go and complete it in your editor!
    ```
5. When you add a unit test in `/test/runtest.jl`, you can try it out by doing
    ```julia
    # if in the package directory:
    ] activate .   
    ] test       

    # from any other location
    ] test HWunconstrained
    ```
6. To submit: 
    ```bash
    cd ~/.julia/dev/HWunconstrained
    git add .   # adds everything. edit if that's not what you want
    git commit -m 'my homework'
    git push
    ```

## License

Please observe that this repo is part of the [Sciences Po CompEcon Organisation](https://github.com/ScPo-CompEcon) and therefore subject to the license detailed at the bottom of [The Syllabus repo](https://github.com/ScPo-CompEcon/Syllabus).