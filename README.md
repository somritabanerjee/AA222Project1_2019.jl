# AA222Project1_2019

## Setup

```
git clone https://github.com/sisl/AA222Project1_2019.jl
cd A222Project1_2019.jl
julia --project -e "using Pkg; Pkg.resolve()"
```

## Usage

Remember to `cd` into the `AA222Project1_2019.jl` directory.

### Simple

#### Python

```
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 1, 10000)' project1.py 
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 2, 10000)' project1.py 
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 3, 10000)' project1.py 
```

#### Julia

```
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 1, 10000)' project1.jl 
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 2, 10000)' project1.jl 
julia --color=yes --project -e 'using AA222Project1_2019; AA222Project1_2019.main("simple", 3, 10000)' project1.jl 
```

