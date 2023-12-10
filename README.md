# MolElec-RS


This programme is meant as an educational project to learn Rust and quantum chemistry. 
As I learned more about Rust and also about quantum chemistry, I decided to rewrite the programme, 
as the design choice and the implementation of the previous programme was not well-designed
to be able to implement all the features I wanted. 

## Dependencies 
If you use `cargo` to compile the programme, then all dependencies should be installed automatically. 
It builds upon the following crates: 
- `ndarray` for the matrix operations
- `ndarray-linalg` for the linear algebra operations
- `rayon` for parallelisation
- `blas-src` for the BLAS backend
- `boys` for a Rust port of the Fortran version to evaluate the Boys function for molecular integrals


